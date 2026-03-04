"""Pedagogical MLP Training Pipeline V2
---------------------------------------
Fills gaps identified in the pedagogical analysis.

  GROUP 7 — Fair Depth Completion  (→ depth_comparison/)
      Missing depths [2, 4, 7, 8, 10, 12, 16] with FAIR config:
      SGD, LR=0.001, NO clipping, NO Adam, random init.
      Same hypers as existing L1, L3, L5, L6.
      Shows true depth wall without optimizer tricks.

  GROUP 8 — Scale Stability: BN+Residual at Scale  (→ scale_stability/)
      H=256, emb=16, ctx=8, depths [4, 8, 12, 16, 20].
      Techniques: kaiming-only vs kaiming+BN+residual.
      SGD, LR=0.001, NO clipping, NO Adam.
      Shows that at scale, BN+residual is ESSENTIAL.
      (Current stability_grid at H=128 showed kaiming alone winning — misleading.)

  GROUP 9 — Data Size Impact  (→ data_size/)
      Same model on 100K, 300K, 500K, 1M, 1.7M chars (Shakespeare + Paul Graham).
      H=256, L=4, ctx=8, emb=16, kaiming+BN+residual, SGD.
      Shows: more data → less overfitting → better val loss.

All groups:
  - SGD only (no Adam, no momentum, no weight decay)
  - NO gradient clipping — lets gradient problems manifest naturally
  - Loss + grad norms logged every 100 steps
  - Text snapshots at milestone steps
  - Compatible with existing regenerate_summaries.py + API

Usage:
    python train_pedagogical_v2.py                    # train all (7, 8, 9)
    python train_pedagogical_v2.py --group 7          # depth completion only
    python train_pedagogical_v2.py --group 8 9        # scale + data experiments
    python train_pedagogical_v2.py --workers 4        # override concurrency
    python train_pedagogical_v2.py --no-skip          # retrain existing models
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
_PAUL_GRAHAM  = _ROOT / "data" / "paul_graham.txt"

OUTPUT_ROOT = CHECKPOINT_DIR / "pedagogical"

# ─────────────────────────────────────────────────────────────────────────────
#  Shared hyper-params for V2 groups (clean SGD, no tricks)
# ─────────────────────────────────────────────────────────────────────────────
V2_LR           = 0.001       # same as existing L1, L3, L5, L6
BATCH_SIZE      = 64
LOG_INTERVAL    = 100
SNAPSHOT_STEPS  = {1000, 5000, 10000, 20000, 50000, 80000, 100000}

# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 7 — Fair Depth Completion
#  Fill missing depths with FAIR config matching L1, L3, L5, L6.
#  SGD, LR=0.001, random init, no BN, no residual, no clipping.
# ─────────────────────────────────────────────────────────────────────────────
G7_N_LAYERS  = [2, 4, 7, 8, 10, 12, 16]
G7_MAX_STEPS = 80_000
G7_EMB_DIM   = 10
G7_HIDDEN    = 128
G7_CTX       = 4

# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 8 — Scale Stability: BN+Residual Shine
#  Bigger models (H=256 and H=512) where kaiming-only fails at depth but BN+res works.
#  SGD, LR=0.001, no clipping.
# ─────────────────────────────────────────────────────────────────────────────
G8_N_LAYERS  = [4, 8, 12, 16, 20]
G8_MAX_STEPS = 100_000
G8_HIDDEN_SIZES = [256, 512]   # Train both sizes to show scaling effect
G8_EMB_DIM   = 16
G8_CTX       = 8

G8_TECHNIQUES = [
    {"name": "kaiming",             "init": "kaiming", "batchnorm": False, "residual": False},
    {"name": "kaiming+BN+residual", "init": "kaiming", "batchnorm": True,  "residual": True},
]

# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 9 — Data Size Impact
#  Same model on progressively larger datasets.
#  H=256, L=4, ctx=8, emb=16, kaiming+BN+residual, SGD.
# ─────────────────────────────────────────────────────────────────────────────
G9_DATA_SIZES = [
    {"label": "100K",  "chars": 100_000,  "source": "shakespeare"},
    {"label": "300K",  "chars": 300_000,  "source": "shakespeare"},
    {"label": "500K",  "chars": 500_000,  "source": "shakespeare"},
    {"label": "1M",    "chars": None,     "source": "shakespeare"},         # full Shakespeare (~1M)
    {"label": "1.7M",  "chars": None,     "source": "combined"},            # Shakespeare + Paul Graham
]
G9_MAX_STEPS = 100_000
G9_HIDDEN    = 256
G9_N_LAYERS  = 4
G9_EMB_DIM   = 16
G9_CTX       = 8

# ─────────────────────────────────────────────────────────────────────────────
#  Concurrency helpers
# ─────────────────────────────────────────────────────────────────────────────
_print_lock = threading.Lock()
_model_lock = threading.Lock()


def tprint(msg: str):
    """Thread-safe print."""
    with _print_lock:
        print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_shakespeare():
    """Load full Shakespeare text, return (text, path)."""
    with open(_SHAKESPEARE, "r", encoding="utf-8") as f:
        return f.read()


def load_combined():
    """Load Shakespeare + Paul Graham combined."""
    texts = []
    for path in [_SHAKESPEARE, _PAUL_GRAHAM]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return "\n\n".join(texts)


def prepare_data(text: str, max_chars: int | None = None):
    """Tokenize text, optionally truncate, split into train/val."""
    if max_chars is not None:
        text = text[:max_chars]
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], tokenizer


def load_default_data():
    """Load Shakespeare with standard split (for Groups 7 & 8)."""
    return prepare_data(load_shakespeare())


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
#  Core training loop (clean SGD, no tricks)
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
    max_steps:    int,
    lr:           float,
    train_data,
    val_data,
    tokenizer,
    label:        str,
    grad_clip_val: float | None = None,   # None = no clipping (default for V2)
    model_seed:   int = SEED,
    dropout_rate:  float = 0.0,
):
    """Train a single model with plain SGD (no Adam, no LR decay, no clipping by default)."""
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
        ).to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    metrics_log = {"train_loss": [], "val_loss": [], "grad_norms": [], "text_snapshots": []}

    expected_uniform = -math.log(1.0 / vocab_size)
    diverged = False
    t0 = time.time()

    clip_tag = f"clip={grad_clip_val}" if grad_clip_val else "no-clip"
    tprint(f"  {label} | {total_params:,} params | SGD lr={lr} {clip_tag} | device={DEVICE}")

    for step in range(1, max_steps + 1):
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

        # Compute grad norm (clip only if grad_clip_val is set)
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
            metrics_log["train_loss"].append({"step": step, "value": round(t_loss, 6)})
            metrics_log["val_loss"].append({"step": step, "value": round(v_loss, 6)})
            metrics_log["grad_norms"].append({"step": step, "value": round(total_grad_norm, 6)})

            if step % 10_000 == 0:
                tprint(
                    f"    [{label}] step {step:>6}/{max_steps}  "
                    f"train={t_loss:.4f}  val={v_loss:.4f}  "
                    f"grad={total_grad_norm:.4f}"
                )

        # Text snapshots at milestone steps
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
            "init_strategy": init_strategy,
            "use_batchnorm": use_batchnorm,
            "use_residual":  use_residual,
            "dropout_rate":  dropout_rate,
            "learning_rate": lr,
            "optimizer_type": "sgd",
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
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Generic concurrent trainer
# ─────────────────────────────────────────────────────────────────────────────

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

    # Write/merge summary (merge with existing if we're adding to depth_comparison)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        # Merge: load existing, update/add new entries by label
        with open(summary_path) as f:
            existing = json.load(f)
        existing_labels = {e["label"] for e in existing}
        for entry in summary:
            if entry["label"] not in existing_labels:
                existing.append(entry)
            else:
                # Replace existing entry with new one
                existing = [e if e["label"] != entry["label"] else entry for e in existing]
        summary = existing

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    tprint(f"\n✅ {group_name} complete — {len(summary)} models in summary.json")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 7 — Fair Depth Completion
# ─────────────────────────────────────────────────────────────────────────────

def train_group7(train_data, val_data, tokenizer, skip_existing=True, max_workers=4):
    """Train missing depth models with fair config (LR=0.001, SGD, random init)."""
    tasks = []
    for i, n_layers in enumerate(G7_N_LAYERS):
        label = f"depth_L{n_layers}"
        tasks.append({
            "label": label,
            "fname": f"{label}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": G7_CTX,
                "emb_dim": G7_EMB_DIM,
                "hidden_size": G7_HIDDEN,
                "num_layers": n_layers,
                "init_strategy": "random",
                "use_batchnorm": False,
                "use_residual": False,
                "max_steps": G7_MAX_STEPS,
                "lr": V2_LR,
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": label,
                "grad_clip_val": None,       # NO clipping
                "model_seed": SEED + 700 + i,
            },
        })
    return _run_concurrent(
        tasks, OUTPUT_ROOT / "depth_comparison",
        skip_existing, max_workers,
        "GROUP 7 — Fair Depth Completion (SGD lr=0.001, random init, no clipping)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 8 — Scale Stability: BN+Residual at Scale
# ─────────────────────────────────────────────────────────────────────────────

def train_group8(train_data, val_data, tokenizer, skip_existing=True, max_workers=4):
    """Show that at H=256 and H=512, BN+residual is essential for deep networks."""
    tasks = []
    task_idx = 0
    for hidden_size in G8_HIDDEN_SIZES:
        for i, n_layers in enumerate(G8_N_LAYERS):
            for j, tech in enumerate(G8_TECHNIQUES):
                label = f"scale_H{hidden_size}_L{n_layers}_{tech['name']}"
                tasks.append({
                    "label": label,
                    "fname": f"{label}.pt",
                    "kwargs": {
                        "vocab_size": tokenizer.vocab_size,
                        "context_size": G8_CTX,
                        "emb_dim": G8_EMB_DIM,
                        "hidden_size": hidden_size,
                        "num_layers": n_layers,
                        "init_strategy": tech["init"],
                        "use_batchnorm": tech["batchnorm"],
                        "use_residual": tech["residual"],
                        "max_steps": G8_MAX_STEPS,
                        "lr": V2_LR,
                        "train_data": train_data,
                        "val_data": val_data,
                        "tokenizer": tokenizer,
                        "label": label,
                        "grad_clip_val": None,       # NO clipping
                        "model_seed": SEED + 800 + task_idx,
                    },
                })
                task_idx += 1
    return _run_concurrent(
        tasks, OUTPUT_ROOT / "scale_stability",
        skip_existing, max_workers,
        f"GROUP 8 — Scale Stability (H={G8_HIDDEN_SIZES}, SGD, no clipping)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 9 — Data Size Impact
# ─────────────────────────────────────────────────────────────────────────────

def train_group9(skip_existing=True, max_workers=3):
    """Same model on different dataset sizes → shows overfitting decreases with more data."""
    shakespeare_text = load_shakespeare()
    combined_text = load_combined()

    tasks = []
    # Each data size needs its own tokenizer+data, so we prepare them upfront
    data_configs = []
    for i, ds in enumerate(G9_DATA_SIZES):
        source_text = combined_text if ds["source"] == "combined" else shakespeare_text
        max_chars = ds["chars"]
        train_d, val_d, tok = prepare_data(source_text, max_chars)
        actual_chars = len(source_text) if max_chars is None else min(max_chars, len(source_text))
        label = f"datasize_{ds['label']}"
        tprint(f"  📊 {label}: {actual_chars:,} chars, vocab={tok.vocab_size}, "
               f"train={len(train_d):,}, val={len(val_d):,}")
        data_configs.append({
            "label": label,
            "train_data": train_d,
            "val_data": val_d,
            "tokenizer": tok,
            "actual_chars": actual_chars,
        })

    for i, dc in enumerate(data_configs):
        tasks.append({
            "label": dc["label"],
            "fname": f"{dc['label']}.pt",
            "kwargs": {
                "vocab_size": dc["tokenizer"].vocab_size,
                "context_size": G9_CTX,
                "emb_dim": G9_EMB_DIM,
                "hidden_size": G9_HIDDEN,
                "num_layers": G9_N_LAYERS,
                "init_strategy": "kaiming",
                "use_batchnorm": True,
                "use_residual": True,
                "max_steps": G9_MAX_STEPS,
                "lr": V2_LR,
                "train_data": dc["train_data"],
                "val_data": dc["val_data"],
                "tokenizer": dc["tokenizer"],
                "label": dc["label"],
                "grad_clip_val": None,       # NO clipping
                "model_seed": SEED + 900 + i,
            },
        })

    return _run_concurrent(
        tasks, OUTPUT_ROOT / "data_size",
        skip_existing, max_workers,
        "GROUP 9 — Data Size Impact (same model, different data amounts)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pedagogical MLP Training V2")
    parser.add_argument("--group", type=int, nargs="+", default=[7, 8, 9],
                        help="Which groups to train (7, 8, 9)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max concurrent training workers")
    parser.add_argument("--no-skip", action="store_true",
                        help="Retrain even if checkpoint exists")
    args = parser.parse_args()

    skip = not args.no_skip
    groups = set(args.group)

    tprint(f"🧪 Pedagogical Training V2")
    tprint(f"   Groups: {sorted(groups)}")
    tprint(f"   Workers: {args.workers}")
    tprint(f"   Skip existing: {skip}")
    tprint(f"   Device: {DEVICE}")
    tprint(f"   Output: {OUTPUT_ROOT}")

    # Groups 7 & 8 share the same data (Shakespeare)
    if groups & {7, 8}:
        tprint(f"\n📚 Loading Shakespeare data from {_SHAKESPEARE} ...")
        train_data, val_data, tokenizer = load_default_data()
        tprint(f"   {len(train_data):,} train + {len(val_data):,} val tokens, "
               f"vocab={tokenizer.vocab_size}")

        if 7 in groups:
            train_group7(train_data, val_data, tokenizer, skip, args.workers)

        if 8 in groups:
            train_group8(train_data, val_data, tokenizer, skip, args.workers)

    # Group 9 loads its own data (multiple sizes)
    if 9 in groups:
        tprint(f"\n📚 Preparing data size experiments...")
        train_group9(skip, args.workers)

    tprint(f"\n{'='*70}")
    tprint(f"🎉 All done! Run `python regenerate_summaries.py` to update manifests.")
    tprint(f"{'='*70}")


if __name__ == "__main__":
    main()
