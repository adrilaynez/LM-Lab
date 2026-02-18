"""
Generic N-Gram Model Engine
---------------------------
Loads precomputed N-Gram statistics and provides inference/visualization.
Supports N=1 (Bigram) to N=5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from api.config import CHECKPOINT_DIR, DEVICE

class NGramModel(nn.Module):
    def __init__(self, vocab_size, context_size=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.model_type = "ngram"
        
        # Load checkpoint
        self.checkpoint_path = CHECKPOINT_DIR / f"ngram_n{context_size}.pt"
        if not self.checkpoint_path.exists():
             raise FileNotFoundError(f"N-Gram checkpoint not found for N={context_size}. Run ngram_precompute.py first.")
        
        print(f"Loading N-Gram Model (N={context_size})...")
        data = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
        
        self.meta = data.get("metadata", {})
        self.vocab = data.get("vocab", [])
        self.vocab_size = len(self.vocab) # Update vocab_size from loaded data
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        
        if data.get("type") == "dense":
            self.tensor = data["tensor"].to(DEVICE)
            self.is_sparse = False
        else:
            self.sparse_data = data["data"]
            self.is_sparse = True
            
        # Hydrate training stats
        self.training_stats = self.meta.get("training_stats", {})
        
    def get_training_stats(self):
        """
        Return training statistics for visualization.
        """
        return self.training_stats
            
    def forward(self, idx):
        """
        Mock forward pass to match API interface.
        input: idx (batch, seq_len)
        output: logits (batch, seq_len, vocab_size)
        
        Note: For N-Gram, we only really care about the LAST position prediction.
        But to be compatible, we can return full sequence if needed, strictly casual.
        """
        # We only implement inference for the last token for now to save compute
        # The API calls this with `idx` and takes `logits[0, -1]` usually.
        
        batch_size, seq_len = idx.shape
        
        # We need to compute logits for the last position
        # Context is the last N tokens of the sequence (excluding the potential next one)
        # Actually `idx` IS the sequence. We want to predict the next token AFTER `idx`?
        # NO. In PyTorch models (like Bigram), forward(idx) returns logits for the sequence
        # where logits[t] is prediction for idx[t+1].
        
        # The API uses:
        # logits, _ = model(idx)
        # last_logits = logits[0, -1] -> this is prediction for the NEXT token after existing sequence.
        
        # So we look at the last `context_size` tokens of `idx` to predict next.
        
        # Get last context
        context_tokens = idx[0, -self.context_size:].tolist()
        
        # Look up probabilities
        probs = self._get_probs(context_tokens)
        
        # Convert to logits (log probs) for compatibility
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        logits = torch.log(probs + epsilon)
        
        # Reshape to (1, 1, vocab_size) to mimic output for the last step
        # The API expects (batch, seq, vocab). We can just return (batch, 1, vocab)
        # and checking code `logits[:, -1, :]` will work if we say seq_len=1.
        # Or we can return full size.
        
        return logits.view(1, 1, -1), None

    def _get_probs(self, context_tokens):
        """
        Retrieve probability distribution for a given context (list of indices).
        """
        # Ensure context is exactly context_size
        if len(context_tokens) < self.context_size:
            # Pad with something? Or just fail?
            # Standard N-gram behavior: backoff?
            # For this strict implementation: simple fallback to uniform or strict.
            # If we don't have enough context, we can't use N-gram of size N.
            # But the user might provide short text.
            # We should pad with separate "start" token or just return 0.
            # Given we trained on full text without explicit start tokens in indices (likely),
            # we might just return uniform if context is too short.
            # OR, we take what we have.
            # Actually precompute script used `text[i : i+N]`.
            # So we strictly need N characters.
            
            # Diagnostic mode: return uniform if not enough context
            return torch.ones(self.vocab_size, device=DEVICE) / self.vocab_size

        if len(context_tokens) > self.context_size:
            context_tokens = context_tokens[-self.context_size:]
            
        ctx_tuple = tuple(context_tokens)
        
        probs = torch.zeros(self.vocab_size, device=DEVICE)
        
        if self.is_sparse:
            if ctx_tuple in self.sparse_data:
                # sparse_data[ctx] is {next_idx: prob}
                next_probs = self.sparse_data[ctx_tuple]
                for idx, p in next_probs.items():
                    probs[idx] = p
            else:
                # Unseen context (Zero probability? Or epsilon?)
                # "Sparse data problem"
                pass 
        else:
             # Dense (Bigram / N=1) -> context is 1 char
             # ctx_tuple is (char_idx,)
             # tensor is [V, V]
             row = ctx_tuple[0]
             if row < self.vocab_size:
                 probs = self.tensor[row]
                 
        return probs

    def get_internals(self, idx):
        """
        Return internals for visualization:
        - active_slice: The specific row/transition stats for the current context.
        """
        if idx.shape[1] < self.context_size:
             # Not enough context
             context_tokens = idx[0].tolist()
        else:
             context_tokens = idx[0, -self.context_size:].tolist()
             
        # "Active Slice"
        # For N=1 (Bigram): slice is vector [V]
        # For N>1: slice is vector [V] from sparse dict
        
        # We compute the probs again to show them
        slice_probs = self._get_probs(context_tokens)
        
        return {
            "active_slice": slice_probs, # Tensor [V]
            "conditioned_on": context_tokens # List[int]
        }
