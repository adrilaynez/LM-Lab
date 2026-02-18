"""
Inference Service
Handles model loading, caching, prediction, and text generation.
"""

import sys
import time
from pathlib import Path
from functools import lru_cache

import torch
import torch.nn.functional as F

# Ensure project root is on sys.path so we can import models/ and utils/
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models import get_model_class
from models.model_registry import MODEL_INFO
from utils.tokenizer import CharTokenizer
from utils.data import load_data
from api.config import CHECKPOINT_DIR, DATA_DIR, DEVICE


# --------------------------------------------------------------------------- #
#  Model loading (cached – loaded once per process)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=8)
def _load_model(model_id: str):
    """
    Load a model from its checkpoint. Returns (model, tokenizer, config, training_info).
    Raises FileNotFoundError if the checkpoint does not exist.
    """
    checkpoint_path = CHECKPOINT_DIR / f"{model_id}_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    training_info = checkpoint.get("training_info", {})

    # Reconstruct tokenizer from checkpoint vocabulary
    tokenizer = CharTokenizer()
    tokenizer.chars = config["chars"]
    tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
    tokenizer.itos = {i: ch for i, ch in enumerate(tokenizer.chars)}
    tokenizer.vocab_size = len(tokenizer.chars)

    # Instantiate and load model weights
    ModelClass = get_model_class(config["model_type"])
    model = ModelClass(config["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    return model, tokenizer, config, training_info


# --------------------------------------------------------------------------- #
#  Public helpers
# --------------------------------------------------------------------------- #

def get_available_model_ids() -> list[str]:
    """Return model IDs that have a checkpoint on disk."""
    return [
        mid for mid in MODEL_INFO
        if (CHECKPOINT_DIR / f"{mid}_checkpoint.pt").exists()
    ]


def get_all_model_ids() -> list[str]:
    """Return all model IDs registered in MODEL_INFO (including unimplemented)."""
    return list(MODEL_INFO.keys())


def model_exists(model_id: str) -> bool:
    return model_id in MODEL_INFO


def checkpoint_exists(model_id: str) -> bool:
    return (CHECKPOINT_DIR / f"{model_id}_checkpoint.pt").exists()


def get_model_detail(model_id: str) -> dict:
    return MODEL_INFO.get(model_id, {})


# --------------------------------------------------------------------------- #
#  Inference
# --------------------------------------------------------------------------- #

def predict(model_id: str, text: str, top_k: int = 10):
    """
    Run a forward pass and return top-k predictions for the next token.
    Returns (predictions_list, full_distribution, tokenizer, elapsed_ms).
    """
    model, tokenizer, config, _ = _load_model(model_id)
    block_size = config.get("block_size", 8)

    t0 = time.perf_counter()

    # Encode input and pad/truncate to block_size
    encoded = tokenizer.encode(text)
    if len(encoded) < block_size:
        encoded = [0] * (block_size - len(encoded)) + encoded
    else:
        encoded = encoded[-block_size:]

    idx = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits, _ = model(idx)

    # Take the last position's logits
    if logits.ndim == 2:
        last_logits = logits[-1]  # shape: (vocab_size,)
    else:
        last_logits = logits[0, -1]

    probs = F.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[0]))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    predictions = [
        {"token": tokenizer.decode([idx_val.item()]), "probability": round(p.item(), 6)}
        for p, idx_val in zip(top_probs, top_indices)
    ]

    full_dist = probs.cpu().tolist()

    return predictions, full_dist, tokenizer, elapsed_ms


def get_internals(model_id: str, text: str, top_k: int = 10):
    """
    Run forward pass + get_internals(), returning both predictions and
    the raw internals dict (tensors).
    """
    model, tokenizer, config, _ = _load_model(model_id)
    block_size = config.get("block_size", 8)

    t0 = time.perf_counter()

    encoded = tokenizer.encode(text)
    if len(encoded) < block_size:
        encoded = [0] * (block_size - len(encoded)) + encoded
    else:
        encoded = encoded[-block_size:]

    idx = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits, _ = model(idx)
        raw_internals = model.get_internals(idx)

    if logits.ndim == 2:
        last_logits = logits[-1]
    else:
        last_logits = logits[0, -1]

    probs = F.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[0]))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    predictions = [
        {"token": tokenizer.decode([idx_val.item()]), "probability": round(p.item(), 6)}
        for p, idx_val in zip(top_probs, top_indices)
    ]

    return predictions, raw_internals, tokenizer, elapsed_ms


def generate(model_id: str, seed_text: str = "", max_length: int = 100, temperature: float = 1.0):
    """
    Auto-regressive text generation.
    """
    model, tokenizer, config, _ = _load_model(model_id)
    block_size = config.get("block_size", 8)

    t0 = time.perf_counter()

    # Start from seed or random
    if seed_text:
        encoded = tokenizer.encode(seed_text)
    else:
        encoded = [0]

    idx = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_length):
            # Crop to block_size
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)

            if logits.ndim == 2:
                last_logits = logits[-1]
            else:
                last_logits = logits[0, -1]

            # Apply temperature
            last_logits = last_logits / temperature
            probs = F.softmax(last_logits, dim=-1)

            # Sample
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx.unsqueeze(0)], dim=1)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    generated = tokenizer.decode(idx[0].tolist())

    return generated, elapsed_ms


# --------------------------------------------------------------------------- #
#  Bigram-Specific Inference (visualization-ready)
# --------------------------------------------------------------------------- #

def run_bigram_inference(text: str, top_k: int = 10) -> dict:
    """
    Full Bigram inference pipeline:
    Registry → BigramModel.forward() + get_internals()
             → Visualization-ready structured data.

    Returns a dict matching BigramInferenceResponse schema.
    """
    model_id = "bigram"
    model, tokenizer, config, training_info = _load_model(model_id)

    t0 = time.perf_counter()

    # ------ 1. Next-token predictions (using last character, bigram-style) ------
    encoded = tokenizer.encode(text)
    last_char_idx = encoded[-1] if encoded else 0
    idx = torch.tensor([[last_char_idx]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits, _ = model(idx)

    # Bigram model: logits shape is (1, vocab_size) after flattening
    last_logits = logits[0] if logits.ndim == 2 else logits[0, -1]
    probs = F.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[0]))

    predictions = [
        {"token": tokenizer.decode([i.item()]), "probability": round(p.item(), 6)}
        for p, i in zip(top_probs, top_indices)
    ]
    full_dist = probs.cpu().tolist()

    # ------ 2. Transition matrix visualization (from get_internals) ------
    dummy = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    internals = model.get_internals(dummy)

    weight_matrix = internals["matrix"]  # raw logits, shape (vocab_size, vocab_size)
    transition_probs = F.softmax(weight_matrix, dim=1).detach().cpu()

    transition_matrix = {
        "shape": list(transition_probs.shape),
        "data": transition_probs.tolist(),
        "row_labels": list(tokenizer.chars),
        "col_labels": list(tokenizer.chars),
    }

    # ------ 3. Training history (from checkpoint metadata) ------
    training = {
        "loss_history": training_info.get("loss_history", []),
        "final_loss": training_info.get("final_loss"),
        "training_steps": training_info.get("training_steps"),
        "learning_rate": config.get("learning_rate"),
        "batch_size": config.get("batch_size"),
        "total_parameters": training_info.get("total_parameters"),
        "trainable_parameters": training_info.get("trainable_parameters"),
        "raw_text_size": training_info.get("raw_text_size"),
        "train_data_size": training_info.get("train_data_size"),
        "val_data_size": training_info.get("val_data_size"),
        "unique_characters": training_info.get("unique_characters"),
    }

    # ------ 4. Architecture info (from model_registry) ------
    detail = MODEL_INFO.get(model_id, {})
    architecture = {
        "name": detail.get("name", "Bigram Model"),
        "description": detail.get("description", ""),
        "type": detail.get("type", ""),
        "complexity": detail.get("complexity", ""),
        "how_it_works": detail.get("how_it_works", []),
        "strengths": detail.get("strengths", []),
        "limitations": detail.get("limitations", []),
        "use_cases": detail.get("use_cases", []),
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "model_id": model_id,
        "model_name": detail.get("name", "Bigram Model"),
        "input": {
            "text": text,
            "token_ids": encoded,
        },
        "predictions": predictions,
        "full_distribution": full_dist,
        "visualization": {
            "transition_matrix": transition_matrix,
            "training": training,
            "architecture": architecture,
        },
        "metadata": {
            "inference_time_ms": round(elapsed_ms, 2),
            "device": DEVICE,
            "vocab_size": tokenizer.vocab_size,
        },
    }


def bigram_generate(start_char: str, num_tokens: int = 100, temperature: float = 1.0) -> dict:
    """
    Bigram text generation with temperature sampling.
    Mirrors bigram_viz.py 'Generate Text' section.
    Returns a dict matching BigramGenerationResponse schema.
    """
    model_id = "bigram"
    model, tokenizer, config, _ = _load_model(model_id)

    if start_char not in tokenizer.stoi:
        raise ValueError(
            f"Character '{start_char}' not found in vocabulary. "
            f"Available: {tokenizer.chars[:20]}..."
        )

    t0 = time.perf_counter()

    g = torch.Generator(device=DEVICE).manual_seed(42)

    output_indices = [tokenizer.stoi[start_char]]
    current_idx = torch.tensor(
        [[tokenizer.stoi[start_char]]], dtype=torch.long, device=DEVICE
    )

    with torch.no_grad():
        for _ in range(num_tokens - 1):
            logits, _ = model(current_idx)
            logits = logits[0, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=g)
            next_char_idx = idx_next.item()
            output_indices.append(next_char_idx)
            current_idx = torch.tensor(
                [[next_char_idx]], dtype=torch.long, device=DEVICE
            )

    generated_text = tokenizer.decode(output_indices)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "model_id": model_id,
        "generated_text": generated_text,
        "length": len(generated_text),
        "temperature": temperature,
        "start_char": start_char,
        "metadata": {
            "inference_time_ms": round(elapsed_ms, 2),
            "device": DEVICE,
            "vocab_size": tokenizer.vocab_size,
        },
    }


def bigram_predict_stepwise(text: str, steps: int = 3) -> dict:
    """
    Multi-step character-by-character prediction from the last
    character of `text`.  Mirrors bigram_viz.py 'Word Prediction' tab.
    Returns a dict matching BigramStepwisePredictionResponse schema.
    """
    model_id = "bigram"
    model, tokenizer, config, _ = _load_model(model_id)

    last_char = text[-1]
    if last_char not in tokenizer.stoi:
        raise ValueError(
            f"Last character '{last_char}' not found in vocabulary. "
            f"Available: {tokenizer.chars[:20]}..."
        )

    t0 = time.perf_counter()

    step_results = []
    current_char = last_char

    with torch.no_grad():
        for step_num in range(1, steps + 1):
            char_idx = tokenizer.stoi[current_char]
            current_idx = torch.tensor(
                [[char_idx]], dtype=torch.long, device=DEVICE
            )

            logits, _ = model(current_idx)
            logits = logits[0, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            next_char = tokenizer.itos[idx_next.item()]
            prob = probs[idx_next].item()

            step_results.append({
                "step": step_num,
                "char": next_char,
                "probability": round(prob, 6),
            })
            current_char = next_char

    final_prediction = "".join(s["char"] for s in step_results)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "model_id": model_id,
        "input_text": text,
        "steps": step_results,
        "final_prediction": final_prediction,
        "metadata": {
            "inference_time_ms": round(elapsed_ms, 2),
            "device": DEVICE,
            "vocab_size": tokenizer.vocab_size,
        },
    }


