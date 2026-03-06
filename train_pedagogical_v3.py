"""Pedagogical MLP Training Pipeline V3
---------------------------------------
GROUP 10 - The Embedding Bottleneck
GROUP 11 - Network Shape (Cylinder vs Pyramid vs Funnel)
GROUP 12 - The Activation Battle
GROUP 13 - Weight Tying
"""

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.config import CHECKPOINT_DIR, DEVICE, SEED
from models.mlp_deep import MLPDeepModel
from utils.tokenizer import CharTokenizer

# ── Data paths ────────────────────────────────────────────────────────────────
_SHAKESPEARE = _ROOT / "data" / "tinyshakespeare_clean.txt"
OUTPUT_ROOT = CHECKPOINT_DIR / "pedagogical"

# ─────────────────────────────────────────────────────────────────────────────
#  Generics
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
LOG_INTERVAL    = 100
SNAPSHOT_STEPS  = {1000, 5000, 10000, 20000, 50000, 80000, 100000}

_print_lock = threading.Lock()
_model_lock = threading.Lock()

def tprint(msg: str):
    """Thread-safe print."""
    with _print_lock:
        print(msg, flush=True)

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_shakespeare():
    with open(_SHAKESPEARE, "r", encoding="utf-8") as f:
        return f.read()

def prepare_data(text: str, max_chars: int | None = None):
    if max_chars is not None:
        text = text[:max_chars]
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], tokenizer

# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation + generation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, data, context_size, batch_size=1024):
    model.eval()
    with torch.no_grad():
        max_start = len(data) - context_size - 1
        if max_start <= 0:
            return float("nan")
        ix = torch.randint(max_start, (batch_size,))
        X = torch.stack([data[i:i + context_size] for i in ix]).to(DEVICE)
        Y = torch.stack([data[i + context_size] for i in ix]).to(DEVICE)
        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y).item()
    model.train()
    return loss

def get_samples(model, tokenizer, context_size, num=3, length=200):
    model.eval()
    prompt = torch.zeros((num, context_size), dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(prompt, length)
        samples = [tokenizer.decode(generated[i]) for i in range(num)]
    model.train()
    return samples

# ─────────────────────────────────────────────────────────────────────────────
#  Core training loop 
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    *,
    vocab_size:   int,
    context_size: int,
    emb_dim:      int,
    hidden_size:  int,
    num_layers:   int,
    init_strategy:str,
    use_batchnorm:bool,
    use_residual: bool,
    hidden_sizes: list[int] | None = None,
    activation_func: str = "tanh",
    tie_weights:  bool = False,
    max_steps:    int,
    optimizer_type: str,     # "sgd" or "adamw"
    lr:           float,
    train_data,
    val_data,
    tokenizer,
    label:        str,
    grad_clip_val: float | None = None,
    model_seed:   int = SEED,
    dropout_rate:  float = 0.0,
):
    with _model_lock:
        set_seed(model_seed)
        model = MLPDeepModel(
            vocab_size=vocab_size,
            context_size=context_size,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            init_strategy=init_strategy,
            use_batchnorm=use_batchnorm,
            use_residual=use_residual,
            dropout_rate=dropout_rate,
            seed=model_seed,
            hidden_sizes=hidden_sizes,
            activation_func=activation_func,
            tie_weights=tie_weights,
        ).to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    metrics_log = {"train_loss": [], "val_loss": [], "grad_norms": [], "text_snapshots": []}
    expected_uniform = -math.log(1.0 / vocab_size)
    diverged = False
    t0 = time.time()

    clip_tag = f"clip={grad_clip_val}" if grad_clip_val else "no-clip"
    tprint(f"  {label} | {total_params:,} params | {optimizer_type.upper()} lr={lr} {clip_tag} | device={DEVICE}")

    for step in range(1, max_steps + 1):
        # LR Cosine Decay for AdamW
        if optimizer_type == "adamw":
            min_lr = lr * 0.1
            decay_ratio = step / max_steps
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            current_lr = min_lr + coeff * (lr - min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        max_start = len(train_data) - context_size - 1
        ix = torch.randint(max_start, (BATCH_SIZE,))
        X = torch.stack([train_data[i:i + context_size] for i in ix]).to(DEVICE)
        Y = torch.stack([train_data[i + context_size] for i in ix]).to(DEVICE)

        model.train()
        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y)

        if torch.isnan(loss) or loss.item() > 300:
            diverged = True
            tprint(f"    ✗ {label} diverged at step {step} (loss={loss.item():.2f})")
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip_val if grad_clip_val is not None else float('inf'),
        ).item()

        optimizer.step()

        if step % LOG_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                t_loss = loss.item()
                v_loss = evaluate(model, val_data, context_size)
                
                # Calculate dead neurons on the current training batch X
                _, _, acts = model(X, return_activations=True)
                dead_frac = model.calculate_dead_neurons(acts)

            metrics_log["train_loss"].append({"step": step, "value": round(t_loss, 6)})
            metrics_log["val_loss"].append({"step": step, "value": round(v_loss, 6)})
            metrics_log["grad_norms"].append({"step": step, "value": round(total_grad_norm, 6)})
            if "dead_neurons" not in metrics_log:
                metrics_log["dead_neurons"] = []
            metrics_log["dead_neurons"].append({"step": step, "value": round(dead_frac, 6)})

            if step % 10_000 == 0:
                tprint(
                    f"    [{label}] step {step:>6}/{max_steps}  "
                    f"train={t_loss:.4f}  val={v_loss:.4f}  "
                    f"grad={total_grad_norm:.4f}  "
                    f"dead={dead_frac*100:.1f}%"
                )

        if step in SNAPSHOT_STEPS and step <= max_steps:
            model.eval()
            with torch.no_grad():
                snap = get_samples(model, tokenizer, context_size, num=1, length=100)
            metrics_log["text_snapshots"].append({"step": step, "text": snap[0]})

    t1 = time.time()

    if diverged:
        final_train = float("inf")
        final_val   = float("inf")
        samples     = ["[diverged]"]
    else:
        model.eval()
        with torch.no_grad():
            final_train = evaluate(model, train_data, context_size)
            final_val   = evaluate(model, val_data, context_size)
            samples     = get_samples(model, tokenizer, context_size)
        tprint(
            f"    ✓ {label} done  train={final_train:.4f}  val={final_val:.4f}  "
            f"time={t1-t0:.1f}s"
        )

    return {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {
            "vocab_size":    vocab_size,
            "context_size":  context_size,
            "emb_dim":       emb_dim,
            "hidden_size":   hidden_size,
            "num_layers":    num_layers,
            "hidden_sizes":  hidden_sizes,
            "activation_func": activation_func,
            "tie_weights":   tie_weights,
            "init_strategy": init_strategy,
            "use_batchnorm": use_batchnorm,
            "use_residual":  use_residual,
            "dropout_rate":  dropout_rate,
            "learning_rate": lr,
            "optimizer_type": optimizer_type,
            "max_steps":     max_steps,
            "batch_size":    BATCH_SIZE,
        },
        "metrics_log": metrics_log,
        "metadata": {
            "label":                label,
            "diverged":             diverged,
            "final_train_loss":     final_train,
            "final_val_loss":       final_val,
            "total_params":         total_params,
            "train_time_sec":       round(t1 - t0, 2),
            "expected_uniform_loss":round(expected_uniform, 6),
            "generated_samples":    samples,
            "vocab":                tokenizer.chars,
            "techniques": {
                "init_strategy": init_strategy,
                "use_batchnorm": use_batchnorm,
                "use_residual":  use_residual,
                "dropout_rate":  dropout_rate,
                "hidden_sizes":  hidden_sizes,
                "activation_func": activation_func,
                "tie_weights":   tie_weights,
            },
        },
    }

def _summary_entry(result: dict) -> dict:
    meta = result["metadata"]
    cfg  = result["config"]
    ml   = result.get("metrics_log", {})
    return {
        "label":              meta["label"],
        "config":             cfg,
        "final_train_loss":   meta["final_train_loss"],
        "final_val_loss":     meta["final_val_loss"],
        "total_params":       meta["total_params"],
        "train_time_sec":     meta["train_time_sec"],
        "diverged":           meta["diverged"],
        "techniques":         meta.get("techniques", {}),
        "generated_samples":  meta["generated_samples"],
        "loss_curve": {
            "train": ml.get("train_loss", []),
            "val":   ml.get("val_loss", []),
        },
        "grad_norms":         ml.get("grad_norms", []),
        "dead_neurons":       ml.get("dead_neurons", []),
        "text_snapshots":     ml.get("text_snapshots", []),
    }

def _run_concurrent(tasks, out_dir, skip_existing, max_workers, group_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    tprint(f"\n{'='*70}")
    tprint(f"{group_name} ({len(tasks)} configs)")
    tprint(f"{'='*70}")

    summary = []
    pending = []

    for task in tasks:
        path = out_dir / task["fname"]
        if skip_existing and path.exists():
            tprint(f"  ⏩ {task['label']} already exists, skipping.")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            summary.append(_summary_entry(ck))
        else:
            pending.append(task)

    if not pending:
        tprint(f"\n✅ {group_name} — all models already exist")
        return summary

    workers = min(max_workers, len(pending))
    tprint(f"  🚀 Training {len(pending)} models with {workers} concurrent workers...")

    def _train_one(task):
        result = train_model(**task["kwargs"])
        path = out_dir / task["fname"]
        torch.save(result, path)
        tprint(f"  💾 Saved → {path.name}")
        return _summary_entry(result)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(_train_one, task): task
            for task in pending
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                entry = future.result()
                summary.append(entry)
            except Exception as e:
                tprint(f"  ✗ {task['label']} FAILED: {e}")
                import traceback
                traceback.print_exc()

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        existing_labels = {e["label"] for e in existing}
        for entry in summary:
            if entry["label"] not in existing_labels:
                existing.append(entry)
            else:
                existing = [e if e["label"] != entry["label"] else entry for e in existing]
        summary = existing

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    tprint(f"\n✅ {group_name} complete — {len(summary)} models in summary.json")
    return summary

# ─────────────────────────────────────────────────────────────────────────────
#  Experiment Groups
# ─────────────────────────────────────────────────────────────────────────────

def train_group10_emb_bottleneck(train_data, val_data, tokenizer, skip_existing=True, max_workers=5):
    """GROUP 10 - Embedding Bottleneck"""
    dims = [2, 4, 8, 32, 128]
    tasks = []
    for i, e in enumerate(dims):
        tasks.append({
            "label": f"emb_bottleneck_E{e}",
            "fname": f"emb_bottleneck_E{e}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": 8,
                "emb_dim": e,
                "hidden_size": 256,
                "num_layers": 4,
                "init_strategy": "kaiming",
                "use_batchnorm": True,
                "use_residual": True,
                "max_steps": 100_000,
                "optimizer_type": "adamw",
                "lr": 0.003,
                "grad_clip_val": 1.0,
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": f"emb_bottleneck_E{e}",
                "model_seed": SEED + 1000,
            }
        })
    return _run_concurrent(tasks, OUTPUT_ROOT / "emb_bottleneck", skip_existing, max_workers, "GROUP 10 - Embedding Bottleneck (AdamW)")

def train_group11_shape(train_data, val_data, tokenizer, skip_existing=True, max_workers=3):
    """GROUP 11 - Network Shape"""
    shapes = [
        {"name": "cylinder", "sizes": [256, 256, 256, 256]},
        {"name": "pyramid",  "sizes": [512, 256, 128, 64]},
        {"name": "funnel",   "sizes": [128, 354, 354, 128]},
    ]
    tasks = []
    for i, s in enumerate(shapes):
        tasks.append({
            "label": f"shape_{s['name']}",
            "fname": f"shape_{s['name']}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": 8,
                "emb_dim": 16,
                "hidden_size": 256, # Dummy
                "num_layers": 4,    # Dummy
                "hidden_sizes": s["sizes"],
                "init_strategy": "kaiming",
                "use_batchnorm": True,
                "use_residual": False, # Disabled for shapes conceptually
                "max_steps": 100_000,
                "optimizer_type": "adamw",
                "lr": 0.003,
                "grad_clip_val": 1.0,
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": f"shape_{s['name']}",
                "model_seed": SEED + 1100,
            }
        })
    return _run_concurrent(tasks, OUTPUT_ROOT / "network_shape", skip_existing, max_workers, "GROUP 11 - Network Shape (AdamW)")

def train_group12_activations(train_data, val_data, tokenizer, skip_existing=True, max_workers=5):
    """GROUP 12 - Activation Battle"""
    acts = ["linear", "sigmoid", "tanh", "relu", "gelu"]
    tasks = []
    for i, a in enumerate(acts):
        tasks.append({
            "label": f"act_{a}",
            "fname": f"act_{a}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": 4,
                "emb_dim": 10,
                "hidden_size": 128,
                "num_layers": 4,
                "activation_func": a,
                "init_strategy": "kaiming",
                "use_batchnorm": False, # Want pure activation performance
                "use_residual": False,
                "max_steps": 80_000,
                "optimizer_type": "sgd", # Pure SGD to show flow breakdown
                "lr": 0.01,
                "grad_clip_val": None,
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": f"act_{a}",
                "model_seed": SEED + 1200,
            }
        })
    return _run_concurrent(tasks, OUTPUT_ROOT / "activation_battle", skip_existing, max_workers, "GROUP 12 - Activation Battle (SGD)")

def train_group13_weight_tying(train_data, val_data, tokenizer, skip_existing=True, max_workers=2):
    """GROUP 13 - Weight Tying"""
    configs = [
        {"name": "untied", "tied": False},
        {"name": "tied",   "tied": True},
    ]
    tasks = []
    for i, c in enumerate(configs):
        tasks.append({
            "label": f"weights_{c['name']}",
            "fname": f"weights_{c['name']}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": 8,
                "emb_dim": 64,      # Must match final hidden size
                "hidden_size": 64,  # Matching emb_dim
                "num_layers": 4,
                "tie_weights": c["tied"],
                "init_strategy": "kaiming",
                "use_batchnorm": True,
                "use_residual": True,
                "max_steps": 80_000,
                "optimizer_type": "adamw",
                "lr": 0.003,
                "grad_clip_val": 1.0,
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": f"weights_{c['name']}",
                "model_seed": SEED + 1300,
            }
        })
    return _run_concurrent(tasks, OUTPUT_ROOT / "weight_tying", skip_existing, max_workers, "GROUP 13 - Weight Tying (AdamW)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, nargs="+", default=[10, 11, 12, 13])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    skip = not args.no_skip
    groups = set(args.group)

    tprint(f"🧪 Pedagogical Training V3")
    tprint(f"   Groups: {sorted(groups)}")
    
    text = load_shakespeare()
    train_data, val_data, tokenizer = prepare_data(text)
    
    if 10 in groups:
        train_group10_emb_bottleneck(train_data, val_data, tokenizer, skip, args.workers)
    if 11 in groups:
        train_group11_shape(train_data, val_data, tokenizer, skip, args.workers)
    if 12 in groups:
        train_group12_activations(train_data, val_data, tokenizer, skip, args.workers)
    if 13 in groups:
        train_group13_weight_tying(train_data, val_data, tokenizer, skip, args.workers)

if __name__ == "__main__":
    main()
