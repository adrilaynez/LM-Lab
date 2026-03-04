"""Pedagogical MLP Training Pipeline (Concurrent)
-------------------------------------------------
Trains six groups of models for interactive visualizations:

  GROUP 1 — Depth Comparison  (§04/§05)
      Depths [2,4,8,12,16], random init, no stability techniques.
      SGD + constant LR + NO clipping → deep models diverge.

  GROUP 2 — Stability Technique Grid  (§06)
      Depths [4,8,12] × {none, kaiming, kaiming+BN, kaiming+BN+residual}
      Same SGD setup → each technique progressively helps.

  GROUP 3 — Big Models  (§08, MLP limitations)
      12 curated configs: size scaling + context scaling.
      AdamW + cosine LR + clipping → all converge but loss plateaus.

  GROUP 4 — Learning Rate Sweep  (§07)
      5 models, same arch, LR from 0.0001 to 0.1.
      Shows optimal LR selection.

  GROUP 5 — Dropout Experiment  (§07)
      3 models: dropout=0 (overfits), 0.2 (optimal), 0.5 (too much).
      Shows regularization effect.

  GROUP 6 — Overtraining Timeline  (§07/§08)
      1 model, 200K steps, text snapshots at milestones.
      Shows quality evolution from gibberish to coherent.

All groups:
  - Loss + gradient norms logged every 100 steps
  - Text snapshots at milestone steps
  - Checkpoints saved to checkpoints/pedagogical/{group}/
  - Concurrent training via ThreadPoolExecutor

Usage:
    python train_pedagogical.py                    # train all groups (1-6)
    python train_pedagogical.py --group 1          # train only group 1
    python train_pedagogical.py --group 4 5 6      # train only new groups
    python train_pedagogical.py --workers 8        # override concurrency
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

# ── Data path — prefer Shakespeare ───────────────────────────────────────────
_SHAKESPEARE = _ROOT / "data" / "tinyshakespeare_clean.txt"
_PAUL_GRAHAM  = _ROOT / "data" / "paul_graham.txt"
DATA_PATH = _SHAKESPEARE if _SHAKESPEARE.exists() else _PAUL_GRAHAM

OUTPUT_ROOT = CHECKPOINT_DIR / "pedagogical"

# ─────────────────────────────────────────────────────────────────────────────
#  Shared hyper-params
# ─────────────────────────────────────────────────────────────────────────────
BASE_EMB_DIM      = 10
BASE_HIDDEN_SIZE  = 128
BASE_CONTEXT_SIZE = 4
BASE_LR           = 0.001       # for AdamW (Group 3)
PEDAGOGICAL_LR    = 0.01        # for SGD (Groups 1 & 2) — amplifies instability
BASE_EPOCHS       = 80_000      # > 50k as requested; enough for convergence
BATCH_SIZE        = 64
LOG_INTERVAL      = 100
GRAD_CLIP         = 5.0         # only used by Group 3 (AdamW)
SNAPSHOT_STEPS    = {1000, 5000, 10000, 20000, 50000, 100000}

# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 1 — Depth Comparison
#  Deep enough for vanishing/exploding gradients to manifest with random init.
#  Uses SGD (no momentum) + constant LR + NO gradient clipping.
# ─────────────────────────────────────────────────────────────────────────────
G1_N_LAYERS  = [2, 4, 8, 12, 16]
G1_MAX_STEPS = BASE_EPOCHS

# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 2 — Stability Grid
#  Same SGD setup as Group 1.  At 8+ layers, random init should diverge,
#  kaiming alone struggles, kaiming+BN works, all three = best.
# ─────────────────────────────────────────────────────────────────────────────
G2_N_LAYERS  = [4, 8, 12]
G2_MAX_STEPS = BASE_EPOCHS

TECHNIQUE_CONFIGS = [
    {"name": "none",               "init": "random",  "batchnorm": False, "residual": False},
    {"name": "kaiming",            "init": "kaiming", "batchnorm": False, "residual": False},
    {"name": "kaiming+BN",         "init": "kaiming", "batchnorm": True,  "residual": False},
    {"name": "kaiming+BN+residual","init": "kaiming", "batchnorm": True,  "residual": True},
]

# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 3 — Big Models (12 curated configs)
#  Two pedagogical stories:
#    A) Size scaling (ctx=8 fixed) → more params barely helps
#    B) Context scaling (h=256, L=4 fixed) → MLP can't use more context
# ─────────────────────────────────────────────────────────────────────────────
G3_CONFIGS = [
    # ── A: Size scaling (ctx=8, emb=16) ──
    {"hidden_size": 128, "n_layers": 2, "context_size": 8,   "emb_dim": 16},
    {"hidden_size": 256, "n_layers": 2, "context_size": 8,   "emb_dim": 16},
    {"hidden_size": 256, "n_layers": 4, "context_size": 8,   "emb_dim": 16},
    {"hidden_size": 512, "n_layers": 4, "context_size": 8,   "emb_dim": 16},
    {"hidden_size": 512, "n_layers": 6, "context_size": 8,   "emb_dim": 16},
    # ── B: Context scaling (h=256, L=4) — ctx=8 shared with series A ──
    {"hidden_size": 256, "n_layers": 4, "context_size": 4,   "emb_dim": 16},
    {"hidden_size": 256, "n_layers": 4, "context_size": 16,  "emb_dim": 16},
    {"hidden_size": 256, "n_layers": 4, "context_size": 32,  "emb_dim": 16},
    {"hidden_size": 256, "n_layers": 4, "context_size": 64,  "emb_dim": 32},
    {"hidden_size": 256, "n_layers": 4, "context_size": 128, "emb_dim": 64},
    # ── "Max" models (push the wall) ──
    {"hidden_size": 512, "n_layers": 6, "context_size": 64,  "emb_dim": 32},
    {"hidden_size": 512, "n_layers": 6, "context_size": 128, "emb_dim": 64},
]
G3_MAX_STEPS = 100_000

# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 4 — Learning Rate Sweep
#  Same model, 5 LRs.  Shows: too low → slow, just right → fast, too high → boom.
# ─────────────────────────────────────────────────────────────────────────────
G4_LRS       = [0.0001, 0.001, 0.005, 0.01, 0.1]
G4_MAX_STEPS = 20_000

# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 5 — Dropout Experiment
#  dropout=0 → overfits, 0.2 → sweet spot, 0.5 → too much regularization.
# ─────────────────────────────────────────────────────────────────────────────
G5_DROPOUT_RATES = [0.0, 0.2, 0.5]
G5_MAX_STEPS     = 100_000

# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 6 — Overtraining Timeline
#  1 model, long training, text snapshots show quality evolution.
# ─────────────────────────────────────────────────────────────────────────────
G6_MAX_STEPS = 200_000


# ─────────────────────────────────────────────────────────────────────────────
#  Concurrency helpers
# ─────────────────────────────────────────────────────────────────────────────
_print_lock = threading.Lock()
_model_lock = threading.Lock()      # serialize model creation (seed + init)


def tprint(msg: str):
    """Thread-safe print."""
    with _print_lock:
        print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], tokenizer


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


def cosine_lr(step: int, max_steps: int, lr_max: float, lr_min: float = 1e-5) -> float:
    if step >= max_steps:
        return lr_min
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / max_steps))


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
    max_steps:    int,
    lr:           float,
    train_data,
    val_data,
    tokenizer,
    label:        str,
    optimizer_type: str = "adamw",   # "adamw" or "sgd"
    use_lr_decay:   bool = True,     # cosine decay (True) or constant LR (False)
    grad_clip_val:  float | None = GRAD_CLIP,  # None = no clipping
    model_seed:     int = SEED,      # per-model seed for thread safety
    dropout_rate:   float = 0.0,
):
    # Serialize model creation to avoid seed race conditions
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

        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    metrics_log = {"train_loss": [], "val_loss": [], "grad_norms": [], "text_snapshots": []}

    expected_uniform = -math.log(1.0 / vocab_size)
    diverged = False
    t0 = time.time()

    opt_tag = f"{optimizer_type.upper()} lr={lr}"
    clip_tag = f"clip={grad_clip_val}" if grad_clip_val else "no-clip"
    drop_tag = f"drop={dropout_rate}" if dropout_rate > 0 else ""
    tag_parts = [t for t in [opt_tag, clip_tag, drop_tag] if t]
    tprint(f"  {label} | {total_params:,} params | {' '.join(tag_parts)} | device={DEVICE}")

    for step in range(1, max_steps + 1):
        # LR schedule: cosine decay or constant
        if use_lr_decay:
            current_lr = cosine_lr(step, max_steps, lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
        else:
            current_lr = lr

        max_start = len(train_data) - context_size - 1
        ix = torch.randint(max_start, (BATCH_SIZE,))
        X = torch.stack([train_data[i:i + context_size] for i in ix]).to(DEVICE)
        Y = torch.stack([train_data[i + context_size] for i in ix]).to(DEVICE)

        model.train()
        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y)

        if torch.isnan(loss) or loss.item() > 25:
            diverged = True
            tprint(f"    ✗ {label} diverged at step {step} (loss={loss.item():.2f})")
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Compute grad norm BEFORE clipping (clip_grad_norm_ returns pre-clip norm)
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
                    f"grad={total_grad_norm:.4f}  lr={current_lr:.2e}"
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
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Generic concurrent trainer
# ─────────────────────────────────────────────────────────────────────────────

def _run_concurrent(tasks, out_dir, skip_existing, max_workers, group_name):
    """Train a list of models concurrently with ThreadPoolExecutor.

    Each task is a dict with:
        "label":  str           — human-readable label
        "fname":  str           — checkpoint filename
        "kwargs": dict          — ALL keyword args for train_model()
    """
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

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    tprint(f"\n✅ {group_name} complete — {len(summary)} models")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 1 — build task list
# ─────────────────────────────────────────────────────────────────────────────

def train_group1(train_data, val_data, tokenizer, skip_existing=True, max_workers=8):
    tasks = []
    for i, n_layers in enumerate(G1_N_LAYERS):
        label = f"depth_L{n_layers}"
        tasks.append({
            "label": label,
            "fname": f"{label}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": BASE_CONTEXT_SIZE,
                "emb_dim": BASE_EMB_DIM,
                "hidden_size": BASE_HIDDEN_SIZE,
                "num_layers": n_layers,
                "init_strategy": "random",
                "use_batchnorm": False,
                "use_residual": False,
                "max_steps": G1_MAX_STEPS,
                "lr": PEDAGOGICAL_LR,
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": label,
                "optimizer_type": "sgd",
                "use_lr_decay": False,
                "grad_clip_val": None,
                "model_seed": SEED + i,
            },
        })
    return _run_concurrent(
        tasks, OUTPUT_ROOT / "depth_comparison",
        skip_existing, max_workers,
        "GROUP 1 — Depth Comparison (SGD, random init, no clipping)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 2 — build task list
# ─────────────────────────────────────────────────────────────────────────────

def train_group2(train_data, val_data, tokenizer, skip_existing=True, max_workers=8):
    tasks = []
    for i, n_layers in enumerate(G2_N_LAYERS):
        for j, tech in enumerate(TECHNIQUE_CONFIGS):
            label = f"L{n_layers}_{tech['name']}"
            tasks.append({
                "label": label,
                "fname": f"{label}.pt",
                "kwargs": {
                    "vocab_size": tokenizer.vocab_size,
                    "context_size": BASE_CONTEXT_SIZE,
                    "emb_dim": BASE_EMB_DIM,
                    "hidden_size": BASE_HIDDEN_SIZE,
                    "num_layers": n_layers,
                    "init_strategy": tech["init"],
                    "use_batchnorm": tech["batchnorm"],
                    "use_residual": tech["residual"],
                    "max_steps": G2_MAX_STEPS,
                    "lr": PEDAGOGICAL_LR,
                    "train_data": train_data,
                    "val_data": val_data,
                    "tokenizer": tokenizer,
                    "label": label,
                    "optimizer_type": "sgd",
                    "use_lr_decay": False,
                    "grad_clip_val": None,
                    "model_seed": SEED + 100 + i * len(TECHNIQUE_CONFIGS) + j,
                },
            })
    return _run_concurrent(
        tasks, OUTPUT_ROOT / "stability_grid",
        skip_existing, max_workers,
        "GROUP 2 — Stability Technique Grid (SGD, no clipping)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 3 — build task list
# ─────────────────────────────────────────────────────────────────────────────

def train_group3(train_data, val_data, tokenizer, skip_existing=True, max_workers=4):
    tasks = []
    for i, cfg in enumerate(G3_CONFIGS):
        hs  = cfg["hidden_size"]
        nl  = cfg["n_layers"]
        ctx = cfg["context_size"]
        emb = cfg["emb_dim"]
        label = f"big_H{hs}_L{nl}_CTX{ctx}_E{emb}"
        tasks.append({
            "label": label,
            "fname": f"{label}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": ctx,
                "emb_dim": emb,
                "hidden_size": hs,
                "num_layers": nl,
                "init_strategy": "kaiming",
                "use_batchnorm": True,
                "use_residual": True,
                "max_steps": G3_MAX_STEPS,
                "lr": BASE_LR,
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": label,
                "model_seed": SEED + 200 + i,
            },
        })
    return _run_concurrent(
        tasks, OUTPUT_ROOT / "big_models",
        skip_existing, max_workers,
        f"GROUP 3 — Big Models ({len(G3_CONFIGS)} configs, AdamW)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 4 — Learning Rate Sweep
#  Same model (4L, kaiming+BN+res), 5 different LRs with AdamW.
# ─────────────────────────────────────────────────────────────────────────────

def train_group4(train_data, val_data, tokenizer, skip_existing=True, max_workers=4):
    tasks = []
    for i, lr_val in enumerate(G4_LRS):
        label = f"lr_sweep_{lr_val}"
        tasks.append({
            "label": label,
            "fname": f"{label}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": 8,
                "emb_dim": 16,
                "hidden_size": 256,
                "num_layers": 4,
                "init_strategy": "kaiming",
                "use_batchnorm": True,
                "use_residual": True,
                "max_steps": G4_MAX_STEPS,
                "lr": lr_val,
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": label,
                "optimizer_type": "adamw",
                "use_lr_decay": False,    # constant LR to isolate the effect
                "grad_clip_val": GRAD_CLIP,
                "model_seed": SEED + 300 + i,
            },
        })
    return _run_concurrent(
        tasks, OUTPUT_ROOT / "lr_sweep",
        skip_existing, max_workers,
        "GROUP 4 — Learning Rate Sweep (AdamW, constant LR)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 5 — Dropout Experiment
#  Same large model, 3 dropout rates.  Trains long enough for overfitting.
# ─────────────────────────────────────────────────────────────────────────────

def train_group5(train_data, val_data, tokenizer, skip_existing=True, max_workers=3):
    tasks = []
    for i, drop in enumerate(G5_DROPOUT_RATES):
        label = f"dropout_{drop}"
        tasks.append({
            "label": label,
            "fname": f"{label}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": 16,
                "emb_dim": 16,
                "hidden_size": 256,
                "num_layers": 6,
                "init_strategy": "kaiming",
                "use_batchnorm": True,
                "use_residual": True,
                "dropout_rate": drop,
                "max_steps": G5_MAX_STEPS,
                "lr": BASE_LR,
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": label,
                "optimizer_type": "adamw",
                "use_lr_decay": True,
                "grad_clip_val": GRAD_CLIP,
                "model_seed": SEED + 400 + i,
            },
        })
    return _run_concurrent(
        tasks, OUTPUT_ROOT / "dropout_experiment",
        skip_existing, max_workers,
        "GROUP 5 — Dropout Experiment (AdamW, kaiming+BN+res)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP 6 — Overtraining Timeline
#  1 model, 200K steps.  Text snapshots show quality evolution.
# ─────────────────────────────────────────────────────────────────────────────

def train_group6(train_data, val_data, tokenizer, skip_existing=True, max_workers=1):
    label = "overtraining_timeline"
    tasks = [{
        "label": label,
        "fname": f"{label}.pt",
        "kwargs": {
            "vocab_size": tokenizer.vocab_size,
            "context_size": 8,
            "emb_dim": 16,
            "hidden_size": 256,
            "num_layers": 4,
            "init_strategy": "kaiming",
            "use_batchnorm": True,
            "use_residual": True,
            "max_steps": G6_MAX_STEPS,
            "lr": BASE_LR,
            "train_data": train_data,
            "val_data": val_data,
            "tokenizer": tokenizer,
            "label": label,
            "optimizer_type": "adamw",
            "use_lr_decay": False,     # constant LR — no decay, so overfitting shows
            "grad_clip_val": GRAD_CLIP,
            "model_seed": SEED + 500,
        },
    }]
    return _run_concurrent(
        tasks, OUTPUT_ROOT / "overtraining_timeline",
        skip_existing, max_workers,
        "GROUP 6 — Overtraining Timeline (200K steps, AdamW)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: compact summary entry (no model weights — for manifest)
# ─────────────────────────────────────────────────────────────────────────────

def _summary_entry(result: dict) -> dict:
    meta = result["metadata"]
    cfg  = result["config"]
    ml   = result.get("metrics_log", {})
    curve = ml.get("val_loss", [])
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
        "val_loss_tail":      curve[-10:] if len(curve) >= 10 else curve,
        "loss_curve": {
            "train": ml.get("train_loss", []),
            "val":   ml.get("val_loss", []),
        },
        "grad_norms":         ml.get("grad_norms", []),
        "text_snapshots":     ml.get("text_snapshots", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train pedagogical MLP groups (concurrent)")
    parser.add_argument(
        "--group", type=int, nargs="+", choices=[1, 2, 3, 4, 5, 6],
        help="Train only these groups (e.g., --group 1 2). Omit to train all.",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-train even if checkpoint already exists.",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Max concurrent workers per group. 0 = auto.",
    )
    args = parser.parse_args()

    skip = not args.no_skip

    # GPU info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"🖥️  GPU: {props.name} — {free/1024**3:.1f}/{total/1024**3:.1f} GB free")
    else:
        print("⚠️  No CUDA GPU — training on CPU (will be slow)")

    print(f"📚 Loading data from {DATA_PATH} ...")
    train_data, val_data, tokenizer = load_data()
    print(
        f"   {len(train_data)+len(val_data):,} chars | "
        f"vocab={tokenizer.vocab_size} | "
        f"train={len(train_data):,} val={len(val_data):,}"
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    groups_to_run = args.group if args.group else [1, 2, 3, 4, 5, 6]
    w = args.workers  # 0 = use per-group defaults

    t_start = time.time()

    if 1 in groups_to_run:
        train_group1(train_data, val_data, tokenizer,
                     skip_existing=skip, max_workers=w or 8)

    if 2 in groups_to_run:
        train_group2(train_data, val_data, tokenizer,
                     skip_existing=skip, max_workers=w or 8)

    if 3 in groups_to_run:
        train_group3(train_data, val_data, tokenizer,
                     skip_existing=skip, max_workers=w or 4)

    if 4 in groups_to_run:
        train_group4(train_data, val_data, tokenizer,
                     skip_existing=skip, max_workers=w or 4)

    if 5 in groups_to_run:
        train_group5(train_data, val_data, tokenizer,
                     skip_existing=skip, max_workers=w or 3)

    if 6 in groups_to_run:
        train_group6(train_data, val_data, tokenizer,
                     skip_existing=skip, max_workers=w or 1)

    elapsed = time.time() - t_start
    print(f"\n🎉 All requested groups complete! Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
