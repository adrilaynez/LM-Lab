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
        hidden_sizes=None,         # [int] for custom shape (overrides hidden_size/num_layers)
        activation_func="tanh",    # "linear" | "sigmoid" | "tanh" | "relu" | "gelu"
        tie_weights=False,         # W_out = W_emb.T
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [hidden_size] * num_layers
        self.hidden_size = hidden_size if hidden_sizes is None else self.hidden_sizes[-1]
        self.num_layers = len(self.hidden_sizes)
        self.init_strategy = init_strategy
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual if hidden_sizes is None else False # Disable residual for custom shapes
        self.dropout_rate = dropout_rate
        self.activation_func = activation_func.lower()
        self.tie_weights = tie_weights
        self.model_type = "mlp_deep"

        torch.manual_seed(seed)

        # Embedding
        self.C = nn.Embedding(vocab_size, emb_dim)

        # Hidden layers
        input_dim = context_size * emb_dim
        layers = []
        for i, h_size in enumerate(self.hidden_sizes):
            in_features = input_dim if i == 0 else self.hidden_sizes[i - 1]
            layers.append(nn.Linear(in_features, h_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_size))
            
            if self.activation_func == "sigmoid":
                layers.append(nn.Sigmoid())
            elif self.activation_func == "tanh":
                layers.append(nn.Tanh())
            elif self.activation_func == "relu":
                layers.append(nn.ReLU())
            elif self.activation_func == "gelu":
                layers.append(nn.GELU())
            elif self.activation_func == "linear":
                layers.append(nn.Identity())
            else:
                raise ValueError(f"Unknown activation: {self.activation_func}")
                
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        self.hidden = nn.Sequential(*layers)

        # Group size for residual forward (Linear + [BN] + Activation + [Dropout])
        self._group_size = 2 + int(use_batchnorm) + int(dropout_rate > 0)

        # For residual: need a projection if input_dim != expected hidden_size
        self.residual_proj = None
        if self.use_residual and input_dim != self.hidden_size:
            self.residual_proj = nn.Linear(input_dim, self.hidden_size)

        # Output layer
        if self.tie_weights:
            assert self.hidden_size == emb_dim, f"For tie_weights, last hidden size ({self.hidden_size}) must equal emb_dim ({emb_dim})"
            # We don't create an output layer, we will use C.weight
            self.output = None
        else:
            self.output = nn.Linear(self.hidden_size, vocab_size)

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
                            # Adjust nonlinearity for kaiming based on activation func
                            if self.activation_func in ["relu", "gelu", "linear", "sigmoid"]:
                                # Kaiming is designed for ReLU, let's use it for gelu and linear as best effort
                                # For sigmoid it's technically Xavier but PyTorch accepts 'sigmoid' in kaiming
                                nl = "relu" if self.activation_func in ["relu", "gelu"] else self.activation_func
                                try:
                                    nn.init.kaiming_normal_(param, nonlinearity=nl)
                                except ValueError:
                                    # Fallback if unsupported (like gelu)
                                    nn.init.kaiming_normal_(param, nonlinearity="relu")
                            else:
                                nn.init.kaiming_normal_(param, nonlinearity="tanh")
                    else:
                        # Random: default PyTorch init (uniform), intentionally bad for deep nets
                        nn.init.normal_(param, 0, 1.0)
                elif "bias" in name:
                    nn.init.zeros_(param)

            # Scale down output layer to prevent initial loss spike
            # Only for proper init — random init should show the full problem
            if self.init_strategy == "kaiming" and self.output is not None:
                self.output.weight.mul_(0.1)

    def forward(self, x, return_activations=False):
        """
        Args:
            x: Input token indices (B, T)
            return_activations: If True, returns (logits, loss, activations_list)
        """
        B, T = x.shape
        emb = self.C(x)  # (B, T, emb_dim)
        emb_flat = emb.view(B, -1)  # (B, T * emb_dim)

        activations = []
        if self.use_residual:
            h = self._forward_residual(emb_flat, activations if return_activations else None)
        else:
            h = emb_flat
            for layer in self.hidden:
                h = layer(h)
                if return_activations and isinstance(layer, (nn.Tanh, nn.ReLU, nn.GELU, nn.Sigmoid, nn.Identity)):
                    activations.append(h.detach())

        if self.tie_weights:
            logits = h @ self.C.weight.T
        else:
            logits = self.output(h)
            
        if return_activations:
            return logits, None, activations
        return logits, None

    def _forward_residual(self, x, activations=None):
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
            
            if activations is not None:
                activations.append(h.detach())

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

    def calculate_dead_neurons(self, activations_list=None):
        """
        Returns fraction of saturated neurons across all hidden layers.
        If using ReLU/GELU: neurons that are strictly <= 0
        If using Tanh: neurons with abs(activation) > 0.99
        If using Sigmoid: neurons with activation < 0.01 or > 0.99
        """
        if not activations_list:
            return 0.0

        total_dead = 0
        total_neurons = 0

        for acts in activations_list:
            # acts shape: (B, T, hidden_size)
            # A neuron is considered dead if it's "dead" across ALL examples and tokens in the batch.
            if self.activation_func in ["relu", "gelu"]:
                # dead if <= 0 for all B, T
                is_dead = (acts <= 0).all(dim=0).all(dim=0)
            elif self.activation_func == "tanh":
                # saturated if abs >= 0.99 for all B, T
                is_dead = (acts.abs() >= 0.99).all(dim=0).all(dim=0)
            elif self.activation_func == "sigmoid":
                # saturated if <= 0.01 or >= 0.99
                is_dead = ((acts <= 0.01) | (acts >= 0.99)).all(dim=0).all(dim=0)
            else:
                is_dead = torch.zeros(acts.shape[-1], dtype=torch.bool, device=acts.device)
            
            total_dead += is_dead.sum().item()
            total_neurons += is_dead.numel()

        if total_neurons == 0:
            return 0.0
        return total_dead / total_neurons

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
