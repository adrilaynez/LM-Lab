"""
Multi-Layer Deep MLP Language Model
------------------------------------
Extension of the standard MLP with configurable:
- Number of hidden layers (1-24)
- Initialization strategy (random vs kaiming)
- BatchNorm (on/off)
- Residual connections (on/off)

Used for the stability technique grid experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from api.config import DEVICE


class MLPDeepModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=3,
        emb_dim=10,
        hidden_size=128,
        num_layers=1,
        init_strategy="kaiming",   # "random" | "kaiming"
        use_batchnorm=False,
        use_residual=False,
        dropout_rate=0.0,
        seed=1337,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_strategy = init_strategy
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        self.model_type = "mlp_deep"

        torch.manual_seed(seed)

        # Embedding
        self.C = nn.Embedding(vocab_size, emb_dim)

        # Hidden layers
        input_dim = context_size * emb_dim
        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_size
            layers.append(nn.Linear(in_features, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Tanh())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        self.hidden = nn.Sequential(*layers)

        # Group size for residual forward (Linear + [BN] + Tanh + [Dropout])
        self._group_size = 2 + int(use_batchnorm) + int(dropout_rate > 0)

        # For residual: need a projection if input_dim != hidden_size
        self.residual_proj = None
        if use_residual and input_dim != hidden_size:
            self.residual_proj = nn.Linear(input_dim, hidden_size)

        # Output layer
        self.output = nn.Linear(hidden_size, vocab_size)

        # Initialize weights
        self._init_weights()

        # Internal state for interpretability
        self.last_layer_activations = {}

    def _init_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    if self.init_strategy == "kaiming":
                        if "residual_proj" in name:
                            nn.init.kaiming_normal_(param, nonlinearity="linear")
                        else:
                            nn.init.kaiming_normal_(param, nonlinearity="tanh")
                    else:
                        # Random: default PyTorch init (uniform), intentionally bad for deep nets
                        nn.init.normal_(param, 0, 1.0)
                elif "bias" in name:
                    nn.init.zeros_(param)

            # Scale down output layer to prevent initial loss spike
            # Only for proper init — random init should show the full problem
            if self.init_strategy == "kaiming":
                self.output.weight.mul_(0.1)

    def forward(self, x):
        B, T = x.shape
        emb = self.C(x)  # (B, T, emb_dim)
        emb_flat = emb.view(B, -1)  # (B, T * emb_dim)

        if self.use_residual:
            h = self._forward_residual(emb_flat)
        else:
            h = self.hidden(emb_flat)

        logits = self.output(h)
        return logits, None

    def _forward_residual(self, x):
        """Forward with residual connections between hidden layers."""
        group_size = self._group_size
        layers = list(self.hidden.children())

        # Project input to hidden_size if needed (for first residual)
        if self.residual_proj is not None:
            x_proj = self.residual_proj(x)
        else:
            x_proj = x

        h = x
        for i in range(self.num_layers):
            start = i * group_size
            group = layers[start:start + group_size]
            
            # The residual should be the projected input for layer 0, or current h otherwise
            residual = x_proj if i == 0 else h
            
            # Apply layer group
            for layer in group:
                h = layer(h)
            
            # Add residual connection
            h = h + residual

        return h

    def get_internals(self, x=None):
        """Return internal states for visualization."""
        if x is not None:
            self.forward(x)

        grad_health = self.get_gradient_flow_health()
        return {
            "embedding_matrix": self.C.weight.detach().cpu(),
            "grad_norms": grad_health["layer_norms"],
            "grad_health": grad_health,
            "dead_neurons": self.calculate_dead_neurons(),
            "weight_stats": self.get_weight_stats(),
        }

    def calculate_dead_neurons(self, threshold=0.99):
        """Returns fraction of saturated neurons across all hidden layers."""
        total_dead = 0
        total_neurons = 0
        group_size = self._group_size
        layers = list(self.hidden.children())

        # Use hooks or just check tanh outputs
        # For simplicity, do a forward pass and check
        # (last forward pass is cached)
        return 0.0  # Placeholder — filled in by training script

    def get_weight_stats(self):
        stats = {}
        for name, param in self.named_parameters():
            p = param.detach().cpu()
            stats[name] = {"mean": p.mean().item(), "std": p.std().item()}
        return stats

    def get_gradient_flow_health(self):
        layer_norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                layer_norms[name] = param.grad.norm().item()
            else:
                layer_norms[name] = 0.0
        total_norm = sum(v**2 for v in layer_norms.values()) ** 0.5
        return {
            "total_grad_norm": total_norm,
            "layer_norms": layer_norms,
            "status": "healthy" if 0.0001 < total_norm < 10.0 else "warning",
        }

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_model_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "total_parameters": total_params,
            "num_layers": self.num_layers,
            "init_strategy": self.init_strategy,
            "use_batchnorm": self.use_batchnorm,
            "use_residual": self.use_residual,
            "dropout_rate": self.dropout_rate,
        }
