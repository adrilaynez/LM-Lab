"""
Transformer Inference Service
=============================
Handles loading GPT grid checkpoints, generation, attention extraction,
and timeline data for the §08–§09 visualizers.
"""

import json
import sys
from pathlib import Path
from functools import lru_cache

import torch
import torch.nn.functional as F

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.gpt import GPTModel
from utils.tokenizer import CharTokenizer
from api.config import CHECKPOINT_DIR, DEVICE

GPT_GRID_DIR = CHECKPOINT_DIR / "gpt_grid"


# --------------------------------------------------------------------------- #
#  Tokenizer (shared across all GPT grid models)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _load_grid_tokenizer() -> CharTokenizer:
    """Load the shared tokenizer for the GPT grid."""
    tok = CharTokenizer()
    json_path = GPT_GRID_DIR / "tokenizer.json"
    pkl_path = GPT_GRID_DIR / "tokenizer.pkl"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok.chars = data["chars"]
        tok.vocab_size = data["vocab_size"]
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
    elif pkl_path.exists():
        tok.load(pkl_path)
    else:
        raise FileNotFoundError("GPT grid tokenizer not found. Run train_gpt_grid.py first.")

    return tok


# --------------------------------------------------------------------------- #
#  Model loading (cached)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=8)
def _load_gpt_model(config_id: str, step: int = None):
    """
    Load a GPT model from the grid.
    If step is None, loads the latest checkpoint (highest step number).
    Returns (model, config_dict).
    """
    config_dir = GPT_GRID_DIR / config_id
    if not config_dir.exists():
        raise FileNotFoundError(f"GPT config '{config_id}' not found in grid.")

    # Find checkpoint
    if step is not None:
        ckpt_path = config_dir / f"step_{step:06d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint step {step} not found for {config_id}")
    else:
        # Find the highest step checkpoint
        ckpt_files = sorted(config_dir.glob("step_*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found for {config_id}")
        ckpt_path = ckpt_files[-1]

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]

    model = GPTModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_blocks=cfg["n_blocks"],
        d_ff=cfg["d_ff"],
        block_size=cfg["block_size"],
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    return model, cfg


# --------------------------------------------------------------------------- #
#  Public API functions
# --------------------------------------------------------------------------- #

def list_gpt_configs() -> list[dict]:
    """List all trained GPT configurations with summary stats."""
    if not GPT_GRID_DIR.exists():
        return []

    configs = []
    for config_dir in sorted(GPT_GRID_DIR.iterdir()):
        if not config_dir.is_dir():
            continue
        summary_path = config_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                configs.append(json.load(f))
    return configs


def get_gpt_timeline(config_id: str) -> dict:
    """
    Get the full training timeline for a config.
    Returns metrics log + checkpoint-level generated samples and attention.
    """
    config_dir = GPT_GRID_DIR / config_id
    if not config_dir.exists():
        raise FileNotFoundError(f"GPT config '{config_id}' not found.")

    # Load metrics log
    metrics_path = config_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Load summary
    summary_path = config_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    # Collect checkpoint data (samples + attention) from each checkpoint file
    checkpoints = []
    for ckpt_path in sorted(config_dir.glob("step_*.pt")):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint_data = {
                "step": ckpt.get("step"),
                "train_loss": ckpt.get("train_loss"),
                "val_loss": ckpt.get("val_loss"),
                "generated_samples": ckpt.get("generated_samples", []),
                "hidden_state_stats": ckpt.get("hidden_state_stats"),
            }
            # Don't include full attention maps in timeline (too large)
            # They're available via the /attention endpoint
            checkpoints.append(checkpoint_data)
        except Exception:
            continue

    return {
        "config_id": config_id,
        "summary": summary,
        "metrics": metrics,
        "checkpoints": checkpoints,
    }


def generate_text(
    config_id: str,
    prompt: str = "",
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = None,
    return_probs: bool = False,
) -> dict:
    """Generate text from a trained GPT model."""
    model, cfg = _load_gpt_model(config_id)
    tokenizer = _load_grid_tokenizer()

    if prompt:
        # Encode prompt, truncate to block_size - 1 to leave room for generation
        ids = tokenizer.encode(prompt)
        max_ctx = cfg["block_size"] - 1
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]
        idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    if return_probs:
        out_idx, all_probs = model.generate_with_probs(
            idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k,
        )
        text = tokenizer.decode(out_idx[0].tolist())
        # Convert probs to char-prob pairs
        prob_dicts = []
        for step_probs in all_probs:
            char_probs = []
            for i, p in enumerate(step_probs):
                if p > 0.001:  # Only include non-negligible probs
                    char_probs.append({"char": tokenizer.itos[i], "prob": round(p, 4)})
            char_probs.sort(key=lambda x: -x["prob"])
            prob_dicts.append(char_probs)

        return {
            "text": text,
            "prompt": prompt,
            "generated": text[len(prompt):] if prompt else text,
            "probabilities": prob_dicts,
        }
    else:
        out_idx = model.generate(
            idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k,
        )
        text = tokenizer.decode(out_idx[0].tolist())
        return {
            "text": text,
            "prompt": prompt,
            "generated": text[len(prompt):] if prompt else text,
        }


def get_attention_maps(config_id: str, text: str = None, step: int = None) -> dict:
    """
    Get attention maps for a given text from a trained GPT model.
    If step is provided, loads checkpoint at that step.
    If text is None, uses the reference attention text from the checkpoint.
    """
    config_dir = GPT_GRID_DIR / config_id
    tokenizer = _load_grid_tokenizer()

    if step is not None:
        # Load from checkpoint (pre-computed)
        ckpt_path = config_dir / f"step_{step:06d}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if ckpt.get("attention_maps") is not None:
                return {
                    "config_id": config_id,
                    "step": step,
                    "input_text": ckpt.get("attention_input_text", ""),
                    "layers": ckpt["attention_maps"],
                }

    # Compute fresh attention maps
    model, cfg = _load_gpt_model(config_id)

    if text is None:
        text = "First, let me tell you about"

    # Truncate to block_size
    text = text[:cfg["block_size"]]
    ids = tokenizer.encode(text)
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    maps = model.get_attention_maps(idx)

    layers = []
    for layer_idx, layer_map in enumerate(maps):
        head_maps = []
        for h in range(layer_map.shape[0]):
            attn = layer_map[h].numpy().tolist()
            head_maps.append(attn)
        layers.append({
            "layer": layer_idx,
            "n_heads": layer_map.shape[0],
            "seq_len": layer_map.shape[1],
            "heads": head_maps,
        })

    return {
        "config_id": config_id,
        "input_text": text,
        "input_chars": list(text),
        "layers": layers,
    }


def get_checkpoint_data(config_id: str, step: int) -> dict:
    """Get full checkpoint data for a specific step."""
    config_dir = GPT_GRID_DIR / config_id
    ckpt_path = config_dir / f"step_{step:06d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint step {step} not found for {config_id}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    return {
        "config_id": config_id,
        "step": ckpt.get("step"),
        "train_loss": ckpt.get("train_loss"),
        "val_loss": ckpt.get("val_loss"),
        "generated_samples": ckpt.get("generated_samples", []),
        "hidden_state_stats": ckpt.get("hidden_state_stats"),
        "attention_input_text": ckpt.get("attention_input_text"),
        "has_attention_maps": ckpt.get("attention_maps") is not None,
    }


def get_available_steps(config_id: str) -> list[int]:
    """List all available checkpoint steps for a config."""
    config_dir = GPT_GRID_DIR / config_id
    if not config_dir.exists():
        return []

    steps = []
    for ckpt_path in sorted(config_dir.glob("step_*.pt")):
        try:
            step = int(ckpt_path.stem.split("_")[1])
            steps.append(step)
        except (IndexError, ValueError):
            continue
    return steps
