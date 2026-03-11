"""
GPT Grid Training Script
=========================
Trains a comprehensive grid of character-level GPT models for the §08–§09 visualizers.
Saves rich checkpoint data at multiple training steps: model weights, loss curves,
generated samples, attention patterns, hidden state norms, and embeddings.

Usage:
    python train_gpt_grid.py              # Train all configs
    python train_gpt_grid.py --config gpt_1b_128d   # Train one specific config
    python train_gpt_grid.py --resume     # Skip already-completed configs

Designed to run overnight on a good GPU. ~25 configs × ~50K steps each.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.gpt import GPTModel
from utils.tokenizer import CharTokenizer

# ============================================================================ #
#  Configuration Grid
# ============================================================================ #

# Reference prompt for all generated samples and attention maps
REFERENCE_PROMPTS = [
    "First, ",                         # Short English prompt
    "",                                # Unconditional generation
]

REFERENCE_ATTENTION_TEXT = "First, let me tell you about the things"  # Fixed text for attention extraction

# Checkpoint steps — save rich data at each of these
CHECKPOINT_STEPS = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000]

# Full configuration grid
# Each config: (config_id, n_blocks, d_model, n_heads, d_ff, block_size, max_steps, purpose)
CONFIGS = [
    # ─── WIDTH SCALING (1 block, varying d_model, ctx=128) ───
    # Shows: more neurons in a single block → diminishing returns (the ceiling)
    ("gpt_1b_32d",    1,   32,  2,   128,  128, 50000, "§08 tiny single-block baseline"),
    ("gpt_1b_64d",    1,   64,  4,   256,  128, 50000, "§08 small single-block"),
    ("gpt_1b_128d",   1,  128,  4,   512,  128, 50000, "§08 medium single-block (reference)"),
    ("gpt_1b_256d",   1,  256,  8,  1024,  128, 50000, "§08 large single-block"),
    ("gpt_1b_512d",   1,  512,  8,  2048,  128, 50000, "§08 xlarge single-block (ceiling demo)"),

    # ─── DEPTH SCALING (128d, varying blocks, ctx=256) ───
    # Shows: stacking blocks breaks the width ceiling
    ("gpt_1b_128d_ctx256",  1,  128, 4,  512, 256, 50000, "§09 depth=1 baseline (longer ctx)"),
    ("gpt_2b_128d",         2,  128, 4,  512, 256, 50000, "§09 depth=2"),
    ("gpt_4b_128d",         4,  128, 4,  512, 256, 50000, "§09 depth=4"),
    ("gpt_6b_128d",         6,  128, 4,  512, 256, 50000, "§09 depth=6"),
    ("gpt_8b_128d",         8,  128, 4,  512, 256, 50000, "§09 depth=8"),
    ("gpt_12b_128d",       12,  128, 4,  512, 256, 50000, "§09 depth=12"),

    # ─── CONTEXT WINDOW SCALING (4 blocks, 128d, varying block_size) ───
    # Shows: longer context → better predictions for long-range dependencies
    ("gpt_4b_128d_ctx32",   4, 128, 4, 512,   32, 50000, "context scaling: 32 chars"),
    ("gpt_4b_128d_ctx64",   4, 128, 4, 512,   64, 50000, "context scaling: 64 chars"),
    ("gpt_4b_128d_ctx128",  4, 128, 4, 512,  128, 50000, "context scaling: 128 chars"),
    ("gpt_4b_128d_ctx256",  4, 128, 4, 512,  256, 50000, "context scaling: 256 chars"),
    ("gpt_4b_128d_ctx512",  4, 128, 4, 512,  512, 40000, "context scaling: 512 chars"),

    # ─── HEAD COUNT SCALING (4 blocks, 128d, varying n_heads, ctx=256) ───
    # Shows: more heads → better specialization (up to a point)
    ("gpt_4b_128d_h1",   4, 128,  1, 512, 256, 50000, "head scaling: 1 head"),
    ("gpt_4b_128d_h2",   4, 128,  2, 512, 256, 50000, "head scaling: 2 heads"),
    ("gpt_4b_128d_h4",   4, 128,  4, 512, 256, 50000, "head scaling: 4 heads (default)"),
    ("gpt_4b_128d_h8",   4, 128,  8, 512, 256, 50000, "head scaling: 8 heads"),
    ("gpt_4b_128d_h16",  4, 128, 16, 512, 256, 50000, "head scaling: 16 heads"),

    # ─── HERO MODELS (bigger, for impressive generation demos) ───
    ("gpt_6b_256d",   6, 256, 8, 1024, 256, 50000, "hero model: 6 blocks 256d"),
    ("gpt_8b_256d",   8, 256, 8, 1024, 256, 50000, "hero model: 8 blocks 256d"),
    ("gpt_12b_256d", 12, 256, 8, 1024, 256, 40000, "hero model: 12 blocks 256d (best quality)"),

    # ─── DROPOUT COMPARISON (4 blocks, 128d, ctx=256) ───
    ("gpt_4b_128d_drop0",   4, 128, 4, 512, 256, 50000, "dropout=0.0 (overfitting demo)"),
    ("gpt_4b_128d_drop03",  4, 128, 4, 512, 256, 50000, "dropout=0.3 (heavy regularization)"),
]

# Dropout overrides for specific configs
DROPOUT_OVERRIDES = {
    "gpt_4b_128d_drop0": 0.0,
    "gpt_4b_128d_drop03": 0.3,
}

# Batch size overrides for large models (to avoid OOM on 8GB VRAM)
BATCH_SIZE_OVERRIDES = {
    "gpt_1b_512d": 32,
    "gpt_8b_256d": 32,
    "gpt_12b_256d": 24,
    "gpt_4b_128d_ctx512": 32,
}

# ============================================================================ #
#  Training Settings
# ============================================================================ #

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
EVAL_INTERVAL = 100       # Evaluate val loss every N steps
EVAL_BATCHES = 20         # Number of batches for val loss estimation
GENERATE_MAX_TOKENS = 250
GENERATE_TEMPERATURE = 0.8
SEED = 1337

# Data
DATA_PATHS = [
    "data/tinyshakespeare_clean.txt",   # ~1MB Shakespeare
    "data/paul_graham.txt",             # ~647KB Paul Graham essays
]


# ============================================================================ #
#  Utilities
# ============================================================================ #

def load_data(paths: list[str]) -> str:
    """Load and concatenate all data files."""
    texts = []
    for p in paths:
        path = Path(p)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
            print(f"  📚 Loaded {path.name}: {len(texts[-1]):,} chars")
        else:
            print(f"  ⚠️  Not found: {path}")
    combined = "\n\n".join(texts)
    print(f"  📊 Total: {len(combined):,} characters")
    return combined


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    """Generate a random batch of (input, target) pairs."""
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"Data too short ({len(data)}) for block_size={block_size}")
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, n_batches=20):
    """Estimate train and val loss over n_batches."""
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total_loss = 0.0
        for _ in range(n_batches):
            xb, yb = get_batch(data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            total_loss += loss.item()
        losses[name] = total_loss / n_batches
    model.train()
    return losses


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, max_tokens, temperature, device):
    """Generate text samples from given prompts."""
    model.eval()
    samples = []
    for prompt in prompts:
        if prompt:
            ids = tokenizer.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
        else:
            # Unconditional: start with a random character
            idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
        text = tokenizer.decode(out[0].tolist())
        samples.append({"prompt": prompt, "generated": text})
    model.train()
    return samples


@torch.no_grad()
def extract_attention_maps(model, tokenizer, text, device):
    """
    Extract attention patterns for a reference text.
    Returns: list of dicts, one per layer, each containing per-head attention matrices.
    """
    model.eval()
    # Truncate text to block_size
    text_truncated = text[:model.block_size]
    ids = tokenizer.encode(text_truncated)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    maps = model.get_attention_maps(idx)

    result = []
    for layer_idx, layer_map in enumerate(maps):
        # layer_map: (n_heads, T, T)
        head_maps = []
        for h in range(layer_map.shape[0]):
            # Downsample if too large (for storage efficiency)
            attn = layer_map[h].numpy().tolist()
            head_maps.append(attn)
        result.append({
            "layer": layer_idx,
            "n_heads": layer_map.shape[0],
            "seq_len": layer_map.shape[1],
            "heads": head_maps,
        })

    model.train()
    return result


@torch.no_grad()
def extract_hidden_state_stats(model, tokenizer, text, device):
    """
    Extract per-layer hidden state statistics for visualization.
    Returns norms and basic stats per layer.
    """
    model.eval()
    text_truncated = text[:model.block_size]
    ids = tokenizer.encode(text_truncated)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    states = model.get_hidden_states(idx)

    stats = []
    for i, state in enumerate(states):
        # state: (T, d_model)
        norms = torch.norm(state, dim=-1)  # (T,)
        stats.append({
            "layer": i,  # 0 = after embedding, 1 = after block 1, etc.
            "mean_norm": norms.mean().item(),
            "std_norm": norms.std().item(),
            "min_norm": norms.min().item(),
            "max_norm": norms.max().item(),
            "mean_val": state.mean().item(),
            "std_val": state.std().item(),
        })

    model.train()
    return stats


def get_lr(step, warmup_steps, max_steps, base_lr):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    # Cosine decay to 10% of base_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.1 + 0.5 * (base_lr - base_lr * 0.1) * (1 + math.cos(math.pi * progress))


# ============================================================================ #
#  Main Training Loop
# ============================================================================ #

def train_one_config(
    config_id: str,
    n_blocks: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    block_size: int,
    max_steps: int,
    purpose: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    tokenizer: CharTokenizer,
    device: str,
    output_dir: Path,
):
    """Train a single GPT configuration and save rich checkpoints."""

    config_dir = output_dir / config_id
    config_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    final_ckpt = config_dir / "final.pt"
    if final_ckpt.exists():
        print(f"  ⏭️  {config_id} already complete, skipping.")
        return

    dropout = DROPOUT_OVERRIDES.get(config_id, 0.1)
    batch_size = BATCH_SIZE_OVERRIDES.get(config_id, BATCH_SIZE)

    print(f"\n{'='*70}")
    print(f"  🚀 Training: {config_id}")
    print(f"     Blocks={n_blocks} d_model={d_model} heads={n_heads} d_ff={d_ff}")
    print(f"     ctx={block_size} dropout={dropout} batch={batch_size} steps={max_steps}")
    print(f"     Purpose: {purpose}")
    print(f"{'='*70}\n")

    torch.manual_seed(SEED)

    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks,
        d_ff=d_ff,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    total_params, trainable_params = model.count_parameters()
    print(f"  📊 Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)

    # Training metrics log (every EVAL_INTERVAL steps)
    metrics_log = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    # Checkpoint steps for this config
    ckpt_steps = set(s for s in CHECKPOINT_STEPS if s <= max_steps)
    ckpt_steps.add(max_steps)  # Always save final

    # Determine which prompts to use, truncating to block_size
    ref_text = REFERENCE_ATTENTION_TEXT[:block_size]
    # Only use reference prompts that fit within block_size
    active_prompts = [p for p in REFERENCE_PROMPTS if len(p) < block_size]

    t0 = time.time()
    best_val_loss = float("inf")

    for step in range(max_steps + 1):
        # ── Learning rate schedule ──
        lr = get_lr(step, WARMUP_STEPS, max_steps, LEARNING_RATE)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Evaluate ──
        if step % EVAL_INTERVAL == 0 or step in ckpt_steps:
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size, device, EVAL_BATCHES)
            metrics_log["steps"].append(step)
            metrics_log["train_loss"].append(round(losses["train"], 4))
            metrics_log["val_loss"].append(round(losses["val"], 4))
            metrics_log["learning_rate"].append(round(lr, 6))

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            print(
                f"  step {step:>6d}/{max_steps} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"lr {lr:.2e} | {steps_per_sec:.1f} steps/s | ETA {eta/60:.1f}m"
            )

        # ── Save checkpoint ──
        if step in ckpt_steps:
            save_checkpoint(
                model, tokenizer, optimizer, step, max_steps,
                metrics_log, active_prompts, ref_text,
                config_id, n_blocks, d_model, n_heads, d_ff, block_size, dropout,
                total_params, purpose, best_val_loss,
                config_dir, device,
            )

        # ── Training step ──
        if step < max_steps:
            xb, yb = get_batch(train_data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    total_time = time.time() - t0
    print(f"\n  ✅ {config_id} complete in {total_time/60:.1f} minutes")
    print(f"     Best val loss: {best_val_loss:.4f}")

    # Save final marker
    summary = {
        "config_id": config_id,
        "total_params": total_params,
        "best_val_loss": round(best_val_loss, 4),
        "final_train_loss": metrics_log["train_loss"][-1] if metrics_log["train_loss"] else None,
        "final_val_loss": metrics_log["val_loss"][-1] if metrics_log["val_loss"] else None,
        "training_time_sec": round(total_time, 1),
        "max_steps": max_steps,
        "purpose": purpose,
        "n_blocks": n_blocks,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "block_size": block_size,
        "dropout": dropout,
    }
    torch.save(summary, final_ckpt)

    # Also save a JSON summary for easy API consumption
    with open(config_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full metrics log as JSON
    with open(config_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f)


def save_checkpoint(
    model, tokenizer, optimizer, step, max_steps,
    metrics_log, prompts, ref_text,
    config_id, n_blocks, d_model, n_heads, d_ff, block_size, dropout,
    total_params, purpose, best_val_loss,
    config_dir, device,
):
    """Save a rich checkpoint with all visualization data."""
    ckpt_path = config_dir / f"step_{step:06d}.pt"

    # Generate samples
    samples = generate_samples(
        model, tokenizer, prompts,
        max_tokens=GENERATE_MAX_TOKENS,
        temperature=GENERATE_TEMPERATURE,
        device=device,
    )

    # Extract attention maps (only if ref_text fits)
    attention_maps = None
    if len(ref_text) >= 10:
        try:
            attention_maps = extract_attention_maps(model, tokenizer, ref_text, device)
        except Exception as e:
            print(f"    ⚠️  Attention extraction failed at step {step}: {e}")

    # Extract hidden state stats
    hidden_stats = None
    if len(ref_text) >= 10:
        try:
            hidden_stats = extract_hidden_state_stats(model, tokenizer, ref_text, device)
        except Exception as e:
            print(f"    ⚠️  Hidden state extraction failed at step {step}: {e}")

    checkpoint = {
        # Model weights
        "model_state_dict": model.state_dict(),

        # Config
        "config": {
            "model_type": "gpt",
            "config_id": config_id,
            "vocab_size": tokenizer.vocab_size,
            "chars": tokenizer.chars,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_blocks": n_blocks,
            "d_ff": d_ff,
            "block_size": block_size,
            "dropout": dropout,
        },

        # Training state
        "step": step,
        "max_steps": max_steps,
        "total_params": total_params,
        "purpose": purpose,

        # Metrics at this step
        "train_loss": metrics_log["train_loss"][-1] if metrics_log["train_loss"] else None,
        "val_loss": metrics_log["val_loss"][-1] if metrics_log["val_loss"] else None,
        "best_val_loss": round(best_val_loss, 4),

        # Full metrics history up to this point
        "metrics_log": {
            "steps": list(metrics_log["steps"]),
            "train_loss": list(metrics_log["train_loss"]),
            "val_loss": list(metrics_log["val_loss"]),
            "learning_rate": list(metrics_log["learning_rate"]),
        },

        # Generated samples
        "generated_samples": samples,

        # Attention patterns (per layer, per head)
        "attention_maps": attention_maps,
        "attention_input_text": ref_text,

        # Hidden state statistics per layer
        "hidden_state_stats": hidden_stats,

        # Embedding matrix snapshot
        "embedding_matrix": model.tok_emb.weight.detach().cpu().numpy().tolist(),
    }

    torch.save(checkpoint, ckpt_path)
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"    💾 Saved checkpoint step {step:,} ({size_mb:.1f} MB)")


# ============================================================================ #
#  Entry Point
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="Train GPT grid for §08–§09 visualizers")
    parser.add_argument("--config", type=str, default=None, help="Train only this config ID")
    parser.add_argument("--resume", action="store_true", help="Skip completed configs")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu)")
    parser.add_argument("--list", action="store_true", help="List all configs and exit")
    args = parser.parse_args()

    # List mode
    if args.list:
        print("\n  Available GPT configurations:\n")
        total_params = 0
        for cfg_id, nb, dm, nh, dff, bs, ms, purpose in CONFIGS:
            # Quick param estimate
            m = GPTModel(vocab_size=70, d_model=dm, n_heads=nh, n_blocks=nb, d_ff=dff, block_size=bs)
            tp, _ = m.count_parameters()
            total_params += tp
            print(f"  {cfg_id:<28s} | {nb:>2d}B {dm:>4d}d {nh:>2d}h ctx={bs:<4d} | {tp:>10,} params | {purpose}")
            del m
        print(f"\n  Total configs: {len(CONFIGS)}")
        print(f"  Total params (sum): {total_params:,}")
        return

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print("\n📚 Loading training data...")
    raw_text = load_data(DATA_PATHS)
    if not raw_text:
        print("❌ No data found! Place .txt files in data/")
        sys.exit(1)

    # Build tokenizer
    tokenizer = CharTokenizer()
    tokenizer.train(raw_text)
    print(f"  🔤 Vocabulary: {tokenizer.vocab_size} characters")
    print(f"     Chars: {''.join(tokenizer.chars[:40])}{'...' if tokenizer.vocab_size > 40 else ''}")

    # Encode and split
    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"  📊 Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    # Output directory
    output_dir = Path("checkpoints") / "gpt_grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer for API use
    tokenizer.save(output_dir / "tokenizer.pkl")
    # Also save as JSON for easy access
    with open(output_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({
            "chars": tokenizer.chars,
            "vocab_size": tokenizer.vocab_size,
            "stoi": tokenizer.stoi,
            "itos": {str(k): v for k, v in tokenizer.itos.items()},
        }, f, ensure_ascii=False, indent=2)

    # Filter configs
    if args.config:
        configs = [(c, nb, dm, nh, dff, bs, ms, p)
                   for c, nb, dm, nh, dff, bs, ms, p in CONFIGS if c == args.config]
        if not configs:
            print(f"❌ Config '{args.config}' not found. Use --list to see available configs.")
            sys.exit(1)
    else:
        configs = CONFIGS

    print(f"\n🏋️ Training {len(configs)} configuration(s)...\n")

    # Train each config
    t_total = time.time()
    completed = 0
    failed = []

    for cfg_id, nb, dm, nh, dff, bs, ms, purpose in configs:
        try:
            train_one_config(
                config_id=cfg_id,
                n_blocks=nb,
                d_model=dm,
                n_heads=nh,
                d_ff=dff,
                block_size=bs,
                max_steps=ms,
                purpose=purpose,
                train_data=train_data,
                val_data=val_data,
                tokenizer=tokenizer,
                device=device,
                output_dir=output_dir,
            )
            completed += 1
        except Exception as e:
            print(f"\n  ❌ FAILED: {cfg_id} — {e}")
            import traceback
            traceback.print_exc()
            failed.append((cfg_id, str(e)))

    total_time = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  🏁 GRID TRAINING COMPLETE")
    print(f"     Completed: {completed}/{len(configs)}")
    print(f"     Failed: {len(failed)}")
    print(f"     Total time: {total_time/3600:.1f} hours")
    if failed:
        print(f"     Failed configs:")
        for cfg_id, err in failed:
            print(f"       - {cfg_id}: {err}")
    print(f"{'='*70}\n")

    # Save grid summary
    grid_summary = {
        "completed": completed,
        "failed": len(failed),
        "total_time_hours": round(total_time / 3600, 2),
        "configs": [],
    }
    for cfg_id, nb, dm, nh, dff, bs, ms, purpose in configs:
        summary_path = output_dir / cfg_id / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                grid_summary["configs"].append(json.load(f))

    with open(output_dir / "grid_summary.json", "w") as f:
        json.dump(grid_summary, f, indent=2)

    print(f"  📄 Grid summary saved to {output_dir / 'grid_summary.json'}")


if __name__ == "__main__":
    main()
