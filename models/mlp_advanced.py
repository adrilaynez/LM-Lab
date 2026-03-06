"""
Advanced Multi-Layer Perceptron (MLP) for Embedding Analysis
------------------------------------------------------------
Implements modern techniques to improve embedding interpretability:
- Weight Tying (Embedding exactly matches output layer weights)
- Orthogonal Initialization option
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from api.config import DEVICE

class MLPAdvancedModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=3,
        emb_dim=10,
        hidden_size=128,
        num_layers=1,
        tie_weights=True,
        orthogonal_init=True,
        seed=1337
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_weights = tie_weights
        self.orthogonal_init = orthogonal_init
        self.model_type = "mlp_advanced"

        torch.manual_seed(seed)

        # 1. Embedding Layer
        self.C = nn.Embedding(vocab_size, emb_dim)

        # 2. Hidden Layers
        self.layers = nn.ModuleList()
        current_dim = context_size * emb_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(current_dim, hidden_size))
            self.layers.append(nn.Tanh())
            current_dim = hidden_size

        # 3. Output Layer
        if self.tie_weights:
            # Para hacer Weight Tying, la última capa oculta debe tener el mismo tamaño que el embedding
            assert self.hidden_size == emb_dim, f"Para tie_weights, hidden_size ({self.hidden_size}) debe igualar a emb_dim ({emb_dim})"
            self.output_layer = None
        else:
            self.output_layer = nn.Linear(hidden_size, vocab_size)

        # Internal states for interpretability
        self.last_h = None

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.orthogonal_init:
                nn.init.orthogonal_(self.C.weight)
            else:
                nn.init.normal_(self.C.weight, 0, 1.0)

            for module in self.layers:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='tanh')
                    nn.init.zeros_(module.bias)

            if not self.tie_weights and self.output_layer is not None:
                nn.init.normal_(self.output_layer.weight, 0, 0.01)
                nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, targets=None):
        B, T = x.shape
        
        emb = self.C(x) # (B, T, emb_dim)
        h = emb.view(B, -1) # (B, T * emb_dim)

        for layer in self.layers:
            h = layer(h)
        
        self.last_h = h

        if self.tie_weights:
            # Logits as dot product between hidden state and embedding weight matrix
            # h is (B, emb_dim), C.weight is (V, emb_dim) -> h @ C.weight.T -> (B, V)
            logits = h @ self.C.weight.T
        else:
            logits = self.output_layer(h)

        loss = None
        if targets is not None:
            # Flatten targets specifically for predicting the next token
            targets_flat = targets.reshape(-1)
            # Repeat logits since we predict the next token continuously if context_size > 1
            # Actually, standard logic expects one next token per context. 
            # We assume x is (B, context_size) and targets is (B)
            # Just matching the training loop format.
            if targets.dim() == 1:
                loss = F.cross_entropy(logits, targets)
            else:
                # If target is (B, T)
                logits_repeated = logits.repeat_interleave(T, dim=0)
                loss = F.cross_entropy(logits_repeated, targets_flat)

        return logits, loss

    def get_internals(self, x=None):
        if x is not None:
            self.forward(x)

        return {
            "embedding_matrix": self.C.weight.detach().cpu(),
            "hidden_activations": self.last_h.detach().cpu() if self.last_h is not None else None,
        }

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
