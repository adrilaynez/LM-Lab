import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import LMEngine

class BigramModel(LMEngine):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        # Weight matrix: vocab_size x vocab_size
        self.W = nn.Parameter(torch.randn((vocab_size, vocab_size)))

    def forward(self, idx, targets=None):
        # Flatten batch and sequence dimensions
        # idx shape: (batch_size, block_size) -> (batch_size * block_size,)
        B, T = idx.shape
        idx_flat = idx.reshape(-1)
        
        # Direct lookup: W[idx] selects row idx
        logits = self.W[idx_flat]  # Shape: (batch_size * block_size, vocab_size)
    
        loss = None
        if targets is not None:
            # Flatten targets to match logits
            targets_flat = targets.reshape(-1)
            # Cross-entropy combines softmax + negative log likelihood
            loss = F.cross_entropy(logits, targets_flat)

        return logits, loss

    def get_internals(self, idx, targets=None):
        # Return weight matrix for visualization
        return {
            "matrix": self.W.detach(),
            "description": "Bigram weight matrix W"
        }