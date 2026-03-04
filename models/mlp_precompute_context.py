"""
Context Window Size Grid Training
----------------------------------
Trains MLP models with varying context_size to demonstrate
how more context improves predictions (with diminishing returns).

Grid: context_size=[1, 2, 3, 5, 8] × emb_dim=10 × hidden_size=128 × lr=0.01
Saves: loss curves, embeddings, generated samples at each snapshot.

Usage:
    python -m models.mlp_precompute_context
"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from api.config import CHECKPOINT_DIR, DATA_PATH, DEVICE, SEED
from models.mlp import MLPModel
from utils.tokenizer import CharTokenizer

# ── Configuration ──
CONTEXT_SIZES = [1, 2, 3, 5, 8]
EMB_DIM = 10
HIDDEN_SIZE = 128
LEARNING_RATE = 0.01
BATCH_SIZE = 32
MAX_STEPS = 50000
SNAPSHOT_STEPS = [0, 1000, 5000, 10000, 20000, 50000]
LOG_INTERVAL = 100

OUTPUT_DIR = CHECKPOINT_DIR / "context_grid"


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


def evaluate(model, data, block_size, batch_size=1000):
    model.eval()
    with torch.no_grad():
        ix = torch.randint(len(data) - block_size, (batch_size,))
        X = torch.stack([data[i : i + block_size] for i in ix]).to(DEVICE)
        Y = torch.stack([data[i + block_size] for i in ix]).to(DEVICE)
        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y).item()
    model.train()
    return loss


def get_samples(model, tokenizer, context_size, num=5, length=40):
    model.eval()
    prompt = torch.zeros((num, context_size), dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(prompt, length)
        samples = [tokenizer.decode(generated[i]) for i in range(num)]
    model.train()
    return samples


def train_context_run(context_size, train_data, val_data, tokenizer):
    set_seed(SEED)
    expected_uniform = -torch.log(torch.tensor(1.0 / tokenizer.vocab_size)).item()

    model = MLPModel(
        vocab_size=tokenizer.vocab_size,
        context_size=context_size,
        emb_dim=EMB_DIM,
        hidden_size=HIDDEN_SIZE,
        seed=SEED,
    ).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    total_params = sum(p.numel() for p in model.parameters())

    metrics_log = {"train_loss": [], "val_loss": []}
    snapshots = {}

    print(f"\n{'='*60}")
    print(f"  Context Size: {context_size} | Params: {total_params:,}")
    print(f"{'='*60}")

    for step in range(MAX_STEPS + 1):
        if step > 0:
            ix = torch.randint(len(train_data) - context_size, (BATCH_SIZE,))
            X = torch.stack([train_data[i : i + context_size] for i in ix]).to(DEVICE)
            Y = torch.stack([train_data[i + context_size] for i in ix]).to(DEVICE)
            logits, _ = model(X)
            loss = F.cross_entropy(logits, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if step % LOG_INTERVAL == 0 or step in SNAPSHOT_STEPS:
            train_loss = evaluate(model, train_data, context_size)
            val_loss = evaluate(model, val_data, context_size)
            metrics_log["train_loss"].append({"step": step, "value": train_loss})
            metrics_log["val_loss"].append({"step": step, "value": val_loss})

        if step in SNAPSHOT_STEPS:
            val_loss = evaluate(model, val_data, context_size)
            train_loss = evaluate(model, train_data, context_size)
            samples = get_samples(model, tokenizer, context_size)
            emb_matrix = model.C.weight.detach().cpu().tolist()

            snapshots[f"step_{step}"] = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "samples": samples,
                "embedding_matrix": emb_matrix,
            }
            print(f"  Step {step:>5d}: train={train_loss:.4f} val={val_loss:.4f}")

    return {
        "config": {
            "context_size": context_size,
            "emb_dim": EMB_DIM,
            "hidden_size": HIDDEN_SIZE,
            "learning_rate": LEARNING_RATE,
            "total_params": total_params,
        },
        "metrics_log": metrics_log,
        "snapshots": snapshots,
        "metadata": {
            "expected_uniform_loss": expected_uniform,
            "final_train_loss": metrics_log["train_loss"][-1]["value"],
            "final_val_loss": metrics_log["val_loss"][-1]["value"],
        },
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_data, val_data, tokenizer = load_data()

    results = {}
    for ctx in CONTEXT_SIZES:
        result = train_context_run(ctx, train_data, val_data, tokenizer)
        results[f"ctx_{ctx}"] = result

        # Save individual checkpoint
        with open(OUTPUT_DIR / f"context_{ctx}.json", "w") as f:
            json.dump(result, f)

    # Save combined summary
    summary = {}
    for ctx in CONTEXT_SIZES:
        r = results[f"ctx_{ctx}"]
        summary[f"ctx_{ctx}"] = {
            "context_size": ctx,
            "total_params": r["config"]["total_params"],
            "final_train_loss": r["metadata"]["final_train_loss"],
            "final_val_loss": r["metadata"]["final_val_loss"],
            "loss_curve": [
                {"step": e["step"], "train": e["value"], "val": v["value"]}
                for e, v in zip(
                    r["metrics_log"]["train_loss"], r["metrics_log"]["val_loss"]
                )
            ],
            "final_samples": r["snapshots"][f"step_{MAX_STEPS}"]["samples"],
        }

    with open(OUTPUT_DIR / "context_grid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Context grid training complete!")
    print(f"   Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
