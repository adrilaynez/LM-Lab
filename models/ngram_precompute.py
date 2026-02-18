"""
N-Gram Precomputation Script
----------------------------
Builds character-level N-Gram statistics (N=1..5) from the specific training corpus.
Saves precomputed probability tensors/dicts to disk for fast loading.

Outputs:
  checkpoints/ngram_n{N}.pt
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict, Counter

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.data import load_data
from utils.tokenizer import CharTokenizer
from api.config import CHECKPOINT_DIR, DATA_DIR

def run_precompute():
    print("🚀 Starting N-Gram Precomputation...")
    
    # 1. Load Data
    data_path = DATA_DIR / "paul_graham.txt"
    try:
        text = load_data(data_path)
    except Exception:
        # Fallback if load_data expects string
        text = load_data(str(data_path))
    
    print(f"📖 Loaded corpus: {len(text)} characters")

    # 2. Build Tokenizer (Character-level)
    # Ensure fixed vocabulary for consistency across all N
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    # itos = {i: ch for i, ch in enumerate(chars)}
    all_chars = chars
    
    print(f"🔤 Vocabulary Size: {vocab_size}")
    
    # Tqdm fallback
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterator, **kwargs):
            return iterator
            
    # 3. Compute N-grams for N=1..5
    MAX_N = 5
    
    for n in range(1, MAX_N + 1):
        print(f"\n📊 Processing N={n}...")
        
        # Container for counts:
        counts = defaultdict(Counter)
        context_size = n
        
        # Collect counts with robust progress
        iterations = len(text) - context_size
        for i in tqdm(range(iterations), desc=f"N={n}"):
            # Context window
            ctx_str = text[i : i + context_size]
            next_char = text[i + context_size]
            
            ctx_idxs = tuple(stoi[c] for c in ctx_str)
            next_idx = stoi[next_char]
            
            counts[ctx_idxs][next_idx] += 1
            
        # --- Compute Rich Statistics ---
        
        # 1. Total observed transitions (sum of all counts)
        total_transitions = sum(sum(next_counts.values()) for next_counts in counts.values())
        
        # 2. Context Statistics
        unique_contexts = len(counts)
        context_space_size = vocab_size ** context_size
        context_utilization = unique_contexts / context_space_size if context_space_size > 0 else 0.0
        
        # 3. Transition Statistics (Sparsity)
        nonzero_transitions = sum(len(next_counts) for next_counts in counts.values())
        potential_observed_transitions = unique_contexts * vocab_size
        
        sparsity = 1.0 - (nonzero_transitions / potential_observed_transitions) if potential_observed_transitions > 0 else 1.0
        
        transition_density = nonzero_transitions / unique_contexts if unique_contexts > 0 else 0.0
        
        print(f"   Matches: {total_transitions:,}")
        print(f"   Unique Contexts: {unique_contexts:,} / {context_space_size:,} ({context_utilization:.2%})")
        print(f"   Sparsity (observed): {sparsity:.2%}")
            
        # Convert to probabilities and save structure
        
        if context_size == 1:
            # Special case: Dense Matrix for Bigram (N=1 context)
            W = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
            for ctx, next_counts in counts.items():
                ctx_idx = ctx[0]
                total = sum(next_counts.values())
                for next_idx, count in next_counts.items():
                    W[ctx_idx, next_idx] = count / total
            
            saved_data = {
                "type": "dense",
                "tensor": W,
                "vocab": all_chars
            }
        else:
            # Sparse case
            # We'll save a dictionary: context_tuple (int, ...) -> {next_idx (int): prob (float)}
            sparse_data = {}
            for ctx, next_counts in counts.items():
                total = sum(next_counts.values())
                probs = {k: v / total for k, v in next_counts.items()}
                sparse_data[ctx] = probs
            
            saved_data = {
                "type": "sparse",
                "data": sparse_data,
                "vocab": all_chars
            }
            
        # Add rich metadata
        saved_data["metadata"] = {
            "context_size": context_size,
            "vocab_size": vocab_size, # Added for explicit retrieval
            "training_stats": {
                "total_tokens": len(text),
                "unique_chars": vocab_size,
                "unique_contexts": unique_contexts,
                "context_space_size": context_space_size,
                "context_utilization": context_utilization,
                "sparsity": sparsity, # Observed sparsity
                "transition_density": transition_density,
                "total_transitions": total_transitions
            }
        }
        
        # Save
        filename = f"ngram_n{context_size}.pt"
        out_path = CHECKPOINT_DIR / filename
        torch.save(saved_data, out_path)
        print(f"✅ Saved {filename}")

if __name__ == "__main__":
    try:
        run_precompute()
    except Exception as e:
        print(f"\n❌ Precomputation Failed: {e}")
        import traceback
        traceback.print_exc()
