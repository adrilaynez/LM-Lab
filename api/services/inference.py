"""
Inference Service
Handles model loading, caching, prediction, and text generation.
"""

import sys
import time
from pathlib import Path
from functools import lru_cache
from fastapi import HTTPException

import torch
import torch.nn.functional as F

# Ensure project root is on sys.path so we can import models/ and utils/
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models import get_model_class
from models.model_registry import MODEL_INFO
from models.ngram import NGramModel
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


@lru_cache(maxsize=8)
def _load_ngram_model(context_size: int):
    """
    Load specific N-Gram model from precomputed checkpoint.
    """
    # Check if checkpoint exists
    if not (CHECKPOINT_DIR / f"ngram_n{context_size}.pt").exists():
        raise FileNotFoundError(f"N-Gram checkpoint for N={context_size} not found.")

    model = NGramModel(vocab_size=0, context_size=context_size) # vocab loaded from ckpt
    model.eval()
    model.to(DEVICE)
    
    # Create tokenizer from model vocabulary
    tokenizer = CharTokenizer()
    tokenizer.chars = model.vocab
    tokenizer.stoi = model.stoi
    tokenizer.itos = model.itos
    tokenizer.vocab_size = len(model.vocab)
    
    return model, tokenizer


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

    # ------ 5. Historical Context ------
    historical_context = {
        "description": "The Bigram model (N=1) represents the simplest Markov chain approximation of language, predicting the next character based solely on the immediate predecessor.",
        "limitations": [
            "Lacks long-range memory (context size = 1)",
            "Cannot capture grammar structure beyond adjacent characters",
            "Treats 'q' -> 'u' and 't' -> 'h' as localized probabilities without understanding words"
        ],
        "modern_evolution": "This strictly local probabilistic approach is the foundation for N-gram models and eventually neural language models, which expanded context to thousands of tokens."
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
            "historical_context": historical_context,
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


# --------------------------------------------------------------------------- #
#  N-Gram & Interpretability
# --------------------------------------------------------------------------- #

from utils.data_explorer import search_dataset




from api.schemas.responses import (
    NGramInferenceResponse, NGramVisualization, NGramTrainingInfo, 
    NGramDiagnostics, ActiveSlice, ModelArchitectureInfo, 
    HistoricalContext, TransitionMatrix, TokenInfo, PredictionResult, InferenceMetadata
)

def run_ngram_inference(text: str, context_size: int, top_k: int = 10) -> dict:
    """
    N-Gram visualization inference with educational safeguards.
    """
    MAX_CONTEXT_SIZE = 5
    if context_size > MAX_CONTEXT_SIZE:
        raise ValueError(f"CONTEXT_TOO_LARGE:{context_size}:{MAX_CONTEXT_SIZE}")

    try:
        model, tokenizer = _load_ngram_model(context_size)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="N-Gram model not initialized. Run precomputation.")

    t0 = time.perf_counter()
    
    # 1. Forward Pass (Prediction)
    encoded = tokenizer.encode(text)
    # Pad if needed? N-gram model handles short context by taking what's available
    # But for strict N-gram, we usually need N previous tokens.
    # Our model._get_probs handles context < N by masking or returning uniform/backoff simulation.
    
    idx = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        logits, _ = model(idx)
        # Logits is [1, 1, V] (last step)
        last_logits = logits[0, -1]
        probs = torch.exp(last_logits) # we returned log_probs
        
    top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[0]))
    
    # 2. Visualization (Active Slice)
    internals = model.get_internals(idx)
    
    context_tokens = []
    if "conditioned_on" in internals:
        c_idxs = internals["conditioned_on"]
        try:
             # c_idxs is list of ints
             context_tokens = [tokenizer.itos.get(i, '?') for i in c_idxs]
        except:
            pass
    
    # Transition Matrix / Active Slice Formatting
    transition_matrix = None
    active_slice = None
    
    if context_size == 1:
        # For N=1 (Bigram), we might want to show the full matrix if requested,
        # but pure N-gram view usually focuses on the active context.
        # However, Bigram has a dedicated "Transformation Matrix" view.
        # For uniformity, we can populate transition_matrix for N=1.
        # But wait, N-gram UI might handle this differently.
        # The schema allows both.
        # If N=1, let's provide the full matrix if feasible, OR just the active slice.
        # Given "Bigram parity", Bigram shows full matrix.
        # Let's see if we can get it.
        # The model has .tensor if dense.
        if hasattr(model, 'tensor'):
             # It's a [V, V] tensor
             prob_matrix = model.tensor.cpu().numpy()
             transition_matrix = {
                "shape": list(prob_matrix.shape),
                "data": prob_matrix.tolist(),
                "row_labels": tokenizer.chars,
                "col_labels": tokenizer.chars
             }
    
    # Construct Active Slice (always relevant)
    if "active_slice" in internals:
        slc_probs = internals["active_slice"] # Tensor [V]
        slc_probs = slc_probs.cpu().numpy()
        
        matrix_data = {
            "shape": [1, len(slc_probs)],
            "data": [slc_probs.tolist()],
            "row_labels": ["Ctx"], # Single row active slice
            "col_labels": tokenizer.chars
        }
        
        active_slice = {
            "context_tokens": context_tokens,
            "matrix": matrix_data,
            "next_token_probs": {
                tokenizer.itos.get(i, str(i)): float(p) 
                for i, p in enumerate(slc_probs) if p > 0.0001 # Filter zero-probs to save space? Or send all?
                # Send all for heatmap
            }
        }
        # Re-populate next_token_probs with ALL for heatmap
        active_slice["next_token_probs"] = {
             tokenizer.itos.get(i, str(i)): float(p) for i, p in enumerate(slc_probs)
        }

    # 3. Training Stats & Diagnostics
    model_stats = model.get_training_stats() # Dict
    
    training = {
        "total_tokens": model_stats.get("total_tokens"),
        "unique_chars": model_stats.get("unique_chars"),
        "unique_contexts": model_stats.get("unique_contexts"),
        "context_space_size": model_stats.get("context_space_size"),
        "context_utilization": model_stats.get("context_utilization"),
        "sparsity": model_stats.get("sparsity"),
        "transition_density": model_stats.get("transition_density")
    }

    # Dynamic Diagnostics
    est_context_space = tokenizer.vocab_size ** context_size
    diagnostics = {
        "vocab_size": tokenizer.vocab_size,
        "context_size": context_size,
        "estimated_context_space": est_context_space,
        "sparsity": model_stats.get("sparsity"),
        "observed_contexts": model_stats.get("unique_contexts"),
        "context_utilization": model_stats.get("context_utilization")
    }
    
    # 4. Architecture Info
    architecture = {
        "name": f"{context_size}-Gram Model",
        "description": f"A probabilistic model that predicts the next character based on the previous {context_size} characters.",
        "type": "Probabilistic / Markov",
        "complexity": "O(1) inference, O(V^N) space",
        "how_it_works": [
            f"Looks at the last {context_size} characters (context).",
            "Consults a pre-calculated table of frequencies from the training text.",
            "Returns the probability distribution for the next character."
        ],
        "strengths": [
            "Extremely fast inference.",
            "Simple to understand and visualize.",
            "Captures local correlations well."
        ],
        "limitations": [
            "Combinatorial explosion: Context space grows exponentially with N.",
            "Sparsity: Most contexts are never seen in training." if context_size > 2 else "Limited context window.",
            "No understanding of semantics or long-range dependencies."
        ],
        "use_cases": [
            "Simple autocomplete.",
            "Language identification.",
            "Baseline for more complex models."
        ]
    }
    
    # 5. Historical Context
    hist_ctx = {
        "description": "N-gram models were the dominant approach in NLP from the 1950s until the rise of neural networks in the 2000s.",
        "limitations": [
            "The 'Curse of Dimensionality': As N increases, the number of possible contexts explodes, requiring impossible amounts of data.",
            "Lack of generalization: Seeing 'cat' doesn't help predict 'dog' in similar contexts without shared representations.",
            "Strict matching: Slight variations in context lead to completely different (or missing) predictions."
        ],
        "modern_evolution": "Neural networks solved the sparsity problem by using distributed representations (embeddings), allowing models to generalize across similar contexts."
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    predictions_list = [
        {"token": tokenizer.decode([i.item()]), "probability": round(p.item(), 6)}
        for p, i in zip(top_probs, top_indices)
    ]
    
    full_dist = probs.tolist()
    
    return {
        "model_id": f"ngram_n{context_size}",
        "model_name": f"{context_size}-Gram",
        "context_size": context_size,
        "input": {
            "text": text,
            "token_ids": encoded
        },
        "predictions": predictions_list,
        "full_distribution": full_dist,
        "visualization": {
            "transition_matrix": transition_matrix,
            "active_slice": active_slice,
            "training": training,
            "diagnostics": diagnostics,
            "architecture": architecture,
            "historical_context": hist_ctx
        },
        "metadata": {
            "inference_time_ms": round(elapsed_ms, 2),
            "device": str(DEVICE),
            "vocab_size": tokenizer.vocab_size,
        }
    }


def run_dataset_lookup(context_tokens: list[str], next_token: str) -> dict:
    """
    Search dataset for context traces.
    Returns DatasetLookupResponse dict.
    """
    return search_dataset(context_tokens, next_token, limit=10)


# --------------------------------------------------------------------------- #
#  N-Gram Stepwise Prediction & Generation
# --------------------------------------------------------------------------- #

def ngram_predict_stepwise(text: str, context_size: int, steps: int = 3, top_k: int = 10) -> dict:
    """
    N-Gram step-by-step character prediction using sliding context windows.
    Generalized version of bigram_predict_stepwise for context_size = N.
    Returns a dict matching NGramStepwisePredictionResponse schema.
    """
    MAX_CONTEXT_SIZE = 5
    if context_size > MAX_CONTEXT_SIZE:
        raise ValueError(f"CONTEXT_TOO_LARGE:{context_size}:{MAX_CONTEXT_SIZE}")

    try:
        model, tokenizer = _load_ngram_model(context_size)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="N-Gram model not initialized. Run precomputation.")

    # Validate that all characters in text are in vocabulary
    for ch in text:
        if ch not in tokenizer.stoi:
            raise ValueError(
                f"Character '{ch}' not in vocabulary. "
                f"Available: {tokenizer.chars[:20]}..."
            )

    t0 = time.perf_counter()

    # Build initial context from the last N characters of text
    encoded = tokenizer.encode(text)
    # If text is shorter than context_size, use what we have (model handles short context)
    context_indices = encoded[-context_size:]

    step_results = []

    with torch.no_grad():
        for step_num in range(1, steps + 1):
            # Get context window as characters for reporting
            context_chars = [tokenizer.itos[i] for i in context_indices]

            # Look up probability distribution
            probs = model._get_probs(context_indices)

            # Handle unseen context (zero vector) → fallback to uniform
            prob_sum = probs.sum().item()
            if prob_sum < 1e-9:
                probs = torch.ones(model.vocab_size, device=DEVICE) / model.vocab_size

            # Top-k predictions for this step
            top_probs_t, top_indices_t = torch.topk(probs, min(top_k, probs.shape[0]))
            top_k_preds = [
                {"token": tokenizer.decode([idx.item()]), "probability": round(p.item(), 6)}
                for p, idx in zip(top_probs_t, top_indices_t)
            ]

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            next_char_idx = idx_next.item()
            next_char = tokenizer.itos[next_char_idx]
            prob = probs[next_char_idx].item()

            step_results.append({
                "step": step_num,
                "char": next_char,
                "probability": round(prob, 6),
                "context_window": context_chars,
                "top_k": top_k_preds,
            })

            # Slide context window: drop oldest, append new
            context_indices = context_indices[1:] + [next_char_idx]
            # If context was shorter than context_size, it grows up to context_size
            if len(context_indices) > context_size:
                context_indices = context_indices[-context_size:]

    final_prediction = "".join(s["char"] for s in step_results)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "model_id": f"ngram_n{context_size}",
        "context_size": context_size,
        "input_text": text,
        "steps": step_results,
        "final_prediction": final_prediction,
        "metadata": {
            "inference_time_ms": round(elapsed_ms, 2),
            "device": str(DEVICE),
            "vocab_size": tokenizer.vocab_size,
        },
    }


def ngram_generate(start_text: str, context_size: int, num_tokens: int = 100, temperature: float = 1.0) -> dict:
    """
    N-Gram text generation with temperature sampling.
    Generalized version of bigram_generate for context_size = N.
    Uses purely lookup-based inference against precomputed probability tables.
    Returns a dict matching NGramGenerationResponse schema.
    """
    MAX_CONTEXT_SIZE = 5
    if context_size > MAX_CONTEXT_SIZE:
        raise ValueError(f"CONTEXT_TOO_LARGE:{context_size}:{MAX_CONTEXT_SIZE}")

    try:
        model, tokenizer = _load_ngram_model(context_size)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="N-Gram model not initialized. Run precomputation.")

    # Validate start_text characters
    for ch in start_text:
        if ch not in tokenizer.stoi:
            raise ValueError(
                f"Character '{ch}' not in vocabulary. "
                f"Available: {tokenizer.chars[:20]}..."
            )

    t0 = time.perf_counter()

    # Initialize output with seed text
    encoded = tokenizer.encode(start_text)
    output_indices = list(encoded)

    with torch.no_grad():
        for _ in range(num_tokens):
            # Take last context_size characters as context
            context_indices = output_indices[-context_size:]

            # Lookup probability distribution
            probs = model._get_probs(context_indices)

            # Handle unseen context (zero vector) → fallback to uniform
            prob_sum = probs.sum().item()
            if prob_sum < 1e-9:
                probs = torch.ones(model.vocab_size, device=DEVICE) / model.vocab_size

            # Apply temperature
            if temperature != 1.0:
                # Convert to log-space, scale, convert back
                log_probs = torch.log(probs + 1e-10)
                log_probs = log_probs / temperature
                probs = F.softmax(log_probs, dim=-1)

            # Sample
            idx_next = torch.multinomial(probs, num_samples=1)
            output_indices.append(idx_next.item())

    generated_text = tokenizer.decode(output_indices)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "model_id": f"ngram_n{context_size}",
        "context_size": context_size,
        "generated_text": generated_text,
        "length": len(generated_text),
        "temperature": temperature,
        "start_text": start_text,
        "metadata": {
            "inference_time_ms": round(elapsed_ms, 2),
            "device": str(DEVICE),
            "vocab_size": tokenizer.vocab_size,
        },
    }

