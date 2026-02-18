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
from tqdm import tqdm

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
    tokenizer = CharTokenizer()
    # Ensure fixed vocabulary for consistency across all N
    all_chars = sorted(list(set(text)))
    tokenizer.chars = all_chars # This might be slightly different from load_data's internal if not careful, but fine for now
    # We should probably trust the tokenizer to build itself from text if we want consistency?
    # Actually, let's reuse the logic from CharTokenizer fit usually, but here we manually set it to be safe 
    # or just let it fit.
    # The Generic CharTokenizer in utils/tokenizer.py likely fits on init or has a fit method. 
    # Let's double check `utils/tokenizer.py` if needed. 
    # For now, let's assume simple standard construction:
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    print(f"🔤 Vocabulary Size: {vocab_size}")
    
    # 3. Compute N-grams for N=1..5
    MAX_N = 5
    
    for n in range(1, MAX_N + 1):
        print(f"\n📊 Processing N={n}...")
        
        # Container for counts:
        # For N=1 (Bigram), context is 1 char -> next char. We want a dense matrix usually?
        # Actually for consistency, let's use a dictionary for all N, 
        # but convert to Dense Tensor for small N if we want, OR just always use Sparse Dict 
        # and convert to Dense on the fly for the "Active Slice".
        # The prompt asks for: "Transition Tensor shape [V^N, V] OR sparse dict"
        # For N=1, V^1 * V is small (96*96).
        # For N=5, 96^5 is huge. Must be sparse.
        
        # We will use a nested dictionary structure: context_tuple -> Counter(next_char_idx)
        # This is essentially a sparse CSR representation logic.
        
        counts = defaultdict(Counter)
        
        # Collect counts
        # Context size is N. So we look at N characters, predict N+1th? 
        # Wait, standard N-gram terminology:
        # "Bigram" (N=2) usually means P(w_i | w_{i-1}). Context size = 1.
        # "Trigram" (N=3) means P(w_i | w_{i-2}, w_{i-1}). Context size = 2.
        # The prompt says: "Build character-level N-gram counts for N=1..5"
        # AND "For N=1 -> return full bigram matrix". This implies N=Context Size.
        # "For N>1 -> compute ACTIVE SLICE: P(next | last N-1 chars)" -> Wait.
        # If N=1 is Bigram, then context size is 1.
        # If the prompt says "N-Gram Engine (N=1..5)", and "N=1 -> return full bigram matrix",
        # it likely means N refers to the CONTEXT SIZE.
        # Let's stick to CONTEXT_SIZE = N.
        
        context_size = n
        
        for i in tqdm(range(len(text) - context_size)):
            # Context window
            ctx_str = text[i : i + context_size]
            next_char = text[i + context_size]
            
            ctx_idxs = tuple(stoi[c] for c in ctx_str)
            next_idx = stoi[next_char]
            
            counts[ctx_idxs][next_idx] += 1
            
        # Convert to probabilities and save
        # We'll save the raw counts or normalized probs? 
        # "Convert counts to probability tensors"
        # To save space, we can save:
        #  { 
        #    context_tuple: { next_idx: count, ... },
        #    ...
        #  }
        # And compute probs on load? Or save probs? 
        # Saving floats takes more space than ints. But load time is faster.
        # Let's save a custom sparse structure.
        
        # Structure to save:
        # keys: list of context tuples (indices)
        # values: list of {next_idx: prob} dicts
        # BUT JSON keys must be strings.
        # Pickle/Torch save handles tuples fine.
        
        # Let's finalize the data structure for the checkpoint:
        # For N=1 (Context=1), we can save a dense tensor [V, V]. 
        # For N>1, we MUST use sparse.
        
        if context_size == 1:
            # Special case: Dense Matrix for Bigram (N=1 context)
            # Shapes: [V, V]
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
                # Normalize and store
                # Only store non-zero transitions
                probs = {k: v / total for k, v in next_counts.items()}
                sparse_data[ctx] = probs
            
            saved_data = {
                "type": "sparse",
                "data": sparse_data,
                "vocab": all_chars
            }
            
        # Add metadata
        saved_data["metadata"] = {
            "context_size": context_size,
            "training_stats": {
                "total_tokens": len(text),
                "unique_chars": vocab_size,
                "unique_contexts": len(counts),
                "context_space_size": vocab_size ** context_size
            }
        }
        
        # Save
        filename = f"ngram_n{context_size}.pt"
        out_path = CHECKPOINT_DIR / filename
        torch.save(saved_data, out_path)
        print(f"✅ Saved {filename} (Contexts: {len(counts)})")

if __name__ == "__main__":
    try:
        run_precompute()
    except Exception as e:
        print(f"\n❌ Precomputation Failed: {e}")
        import traceback
        traceback.print_exc()
