"""
Stability Technique Grid Training
-----------------------------------
Trains deep MLP models to demonstrate that deep networks
only become trainable when combining all stability techniques.

Grid: num_layers=[1,2,3,4] × init=[random,kaiming] × batchnorm=[off,on] × residual=[off,on]
Fixed: emb_dim=10, hidden_size=128, context_size=3, lr=0.01

Usage:
    python -m models.mlp_precompute_stability
"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from api.config import CHECKPOINT_DIR, DATA_PATH, DEVICE, SEED
from models.mlp_deep import MLPDeepModel
from utils.tokenizer import CharTokenizer

# ── Configuration ──
NUM_LAYERS_LIST = [1, 2, 3, 4]
INIT_STRATEGIES = ["random", "kaiming"]
BATCHNORM_OPTIONS = [False, True]
RESIDUAL_OPTIONS = [False, True]

EMB_DIM = 10
HIDDEN_SIZE = 128
CONTEXT_SIZE = 3
LEARNING_RATE = 0.01
BATCH_SIZE = 32
MAX_STEPS = 20000
LOG_INTERVAL = 100

OUTPUT_DIR = CHECKPOINT_DIR / "stability_grid"


def set_seed(seed):
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


def evaluate(model, data, block_size, batch_size=500):
    model.eval()
    with torch.no_grad():
        ix = torch.randint(len(data) - block_size, (batch_size,))
        X = torch.stack([data[i : i + block_size] for i in ix]).to(DEVICE)
        Y = torch.stack([data[i + block_size] for i in ix]).to(DEVICE)
        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y).item()
    model.train()
    return loss


def get_samples(model, tokenizer, context_size, num=3, length=30):
    model.eval()
    prompt = torch.zeros((num, context_size), dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(prompt, length)
        samples = [tokenizer.decode(generated[i]) for i in range(num)]
    model.train()
    return samples


def train_stability_run(cfg, train_data, val_data, tokenizer):
    set_seed(SEED)
    expected_uniform = -torch.log(torch.tensor(1.0 / tokenizer.vocab_size)).item()

    model = MLPDeepModel(
        vocab_size=tokenizer.vocab_size,
        context_size=CONTEXT_SIZE,
        emb_dim=EMB_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=cfg["num_layers"],
        init_strategy=cfg["init"],
        use_batchnorm=cfg["batchnorm"],
        use_residual=cfg["residual"],
        seed=SEED,
    ).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    total_params = sum(p.numel() for p in model.parameters())

    metrics_log = {"train_loss": [], "val_loss": [], "grad_norms": []}

    label = f"L{cfg['num_layers']}_{cfg['init']}_BN{int(cfg['batchnorm'])}_Res{int(cfg['residual'])}"
    print(f"  {label} ({total_params:,} params) ...", end=" ", flush=True)

    diverged = False
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        ix = torch.randint(len(train_data) - CONTEXT_SIZE, (BATCH_SIZE,))
        X = torch.stack([train_data[i : i + CONTEXT_SIZE] for i in ix]).to(DEVICE)
        Y = torch.stack([train_data[i + CONTEXT_SIZE] for i in ix]).to(DEVICE)

        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y)

        if torch.isnan(loss) or loss.item() > 20:
            diverged = True
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step % LOG_INTERVAL == 0:
            train_loss = loss.item()
            val_loss = evaluate(model, val_data, CONTEXT_SIZE)
            metrics_log["train_loss"].append({"step": step, "value": train_loss})
            metrics_log["val_loss"].append({"step": step, "value": val_loss})

            # Track gradient norms
            total_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad += p.grad.norm().item() ** 2
            metrics_log["grad_norms"].append(
                {"step": step, "value": total_grad**0.5}
            )

    t1 = time.time()

    if diverged:
        final_train = float("inf")
        final_val = float("inf")
        samples = ["[diverged]"]
        print(f"DIVERGED at step {step}")
    else:
        final_train = evaluate(model, train_data, CONTEXT_SIZE)
        final_val = evaluate(model, val_data, CONTEXT_SIZE)
        samples = get_samples(model, tokenizer, CONTEXT_SIZE)
        print(f"loss={final_val:.3f} ({t1-t0:.1f}s)")

    return {
        "config": {
            "num_layers": cfg["num_layers"],
            "init": cfg["init"],
            "batchnorm": cfg["batchnorm"],
            "residual": cfg["residual"],
            "emb_dim": EMB_DIM,
            "hidden_size": HIDDEN_SIZE,
            "total_params": total_params,
        },
        "metrics_log": metrics_log,
        "metadata": {
            "diverged": diverged,
            "final_train_loss": final_train,
            "final_val_loss": final_val,
            "expected_uniform_loss": expected_uniform,
            "train_time_sec": t1 - t0,
            "samples": samples,
        },
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_data, val_data, tokenizer = load_data()

    all_results = []
    for num_layers in NUM_LAYERS_LIST:
        print(f"\n── Layers: {num_layers} ──")
        for init in INIT_STRATEGIES:
            for bn in BATCHNORM_OPTIONS:
                for res in RESIDUAL_OPTIONS:
                    cfg = {
                        "num_layers": num_layers,
                        "init": init,
                        "batchnorm": bn,
                        "residual": res,
                    }
                    result = train_stability_run(cfg, train_data, val_data, tokenizer)
                    all_results.append(result)

    # Build summary matrix
    summary = {
        "grid_axes": {
            "rows": "num_layers",
            "row_values": NUM_LAYERS_LIST,
            "columns": "technique",
            "column_values": [
                "random",
                "kaiming",
                "kaiming+BN",
                "kaiming+BN+residual",
            ],
        },
        "cells": [],
    }

    for r in all_results:
        cfg = r["config"]
        # Determine technique column
        if cfg["init"] == "random":
            technique = "random"
        elif not cfg["batchnorm"] and not cfg["residual"]:
            technique = "kaiming"
        elif cfg["batchnorm"] and not cfg["residual"]:
            technique = "kaiming+BN"
        elif cfg["batchnorm"] and cfg["residual"]:
            technique = "kaiming+BN+residual"
        else:
            technique = f"{cfg['init']}_BN{int(cfg['batchnorm'])}_Res{int(cfg['residual'])}"

        summary["cells"].append(
            {
                "num_layers": cfg["num_layers"],
                "technique": technique,
                "final_val_loss": r["metadata"]["final_val_loss"],
                "diverged": r["metadata"]["diverged"],
                "total_params": cfg["total_params"],
                "samples": r["metadata"]["samples"],
                "loss_curve": [
                    {"step": e["step"], "value": e["value"]}
                    for e in r["metrics_log"]["val_loss"]
                ],
            }
        )

    with open(OUTPUT_DIR / "stability_grid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results
    with open(OUTPUT_DIR / "stability_grid_full.json", "w") as f:
        json.dump(all_results, f)

    print("\n✅ Stability grid training complete!")
    print(f"   Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
