"""
Multi-Layer Perceptron (MLP) Language Model
-------------------------------------------
Character-level MLP inspired by Bengio et al. 2003 and Karpathy's 'makemore'.
Architecture: Embedding -> Hidden (Linear + Tanh) -> Output (Linear)

Key features:
- Configurable embedding dimension and hidden size
- Strict reproducibility via manual_seed
- Kaiming initialization for Tanh saturation prevention
- Interpretability hooks (dead neurons, activation stats)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from api.config import DEVICE

class MLPModel(nn.Module):
    def __init__(self, vocab_size, context_size=3, emb_dim=10, hidden_size=200, seed=1337):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.model_type = "mlp"

        # Strict reproducibility
        torch.manual_seed(seed)
        
        # --- Architecture ---
        # 1. Embedding Layer: C (vocab_size, emb_dim)
        self.C = nn.Embedding(vocab_size, emb_dim)
        
        # 2. Hidden Layer: W1, b1
        self.input_dim = context_size * emb_dim
        self.W1 = nn.Parameter(torch.randn(self.input_dim, hidden_size))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        
        # 3. Output Layer: W2, b2
        self.W2 = nn.Parameter(torch.randn(hidden_size, vocab_size))
        self.b2 = nn.Parameter(torch.zeros(vocab_size))
        
        # Internal state for interpretability
        self.last_h_pre = None
        self.last_h = None
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """
        Custom initialization for Tanh activation checks.
        - W1: Kaiming-style for Tanh: (5/3) / sqrt(fan_in)
        - W2: Scale down to 0.1 to reduce initial loss spike
        - b2: Zero init
        """
        gain = 5/3 # Recommended for Tanh
        std1 = gain / (self.input_dim ** 0.5)
        
        with torch.no_grad():
            self.W1.normal_(0, std1)
            self.b1.zero_()
            
            # Output layer scaled down to ensure initial uniform probability
            # and prevent large initial loss spikes
            self.W2.normal_(0, 1.0) # Normal first
            self.W2 *= 0.1 # Then scale by 0.1 as requested
            self.b2.zero_()

    def forward(self, x):
        """
        Forward pass.
        x: (batch_size, context_size)
        """
        B, T = x.shape
        
        # 1. Embed
        emb = self.C(x) # (B, T, emb_dim)
        emb_flat = emb.view(B, -1) # (B, T * emb_dim)
        
        # 2. Hidden Layer
        h_pre = emb_flat @ self.W1 + self.b1
        h = torch.tanh(h_pre)
        
        # Store for interpretability
        self.last_h_pre = h_pre
        self.last_h = h
        
        # 3. Output
        logits = h @ self.W2 + self.b2 # (B, vocab_size)
        
        return logits, None
        
    def get_internals(self, x=None):
        """
        Return internal states for visualization/interpretability.
        """
        if x is not None:
             self.forward(x)
             
        # Compute gradient flow health
        grad_health = self.get_gradient_flow_health()

        return {
            "embedding_matrix": self.C.weight.detach().cpu(),
            "hidden_activations": self.last_h.detach().cpu() if self.last_h is not None else None,
            "hidden_preactivations": self.last_h_pre.detach().cpu() if self.last_h_pre is not None else None,
            "W1": self.W1.detach().cpu(),
            "W2": self.W2.detach().cpu(),
            "grad_norms": grad_health["layer_norms"],
            "grad_health": grad_health,
            "dead_neurons": self.calculate_dead_neurons() if self.last_h is not None else 0.0,
            "weight_stats": self.get_weight_stats(),
            "activation_stats": self.get_activation_stats(),
        }
        
    def calculate_dead_neurons(self, threshold=0.99):
        """
        Returns percentage of neurons that are saturated (dead) in the last forward pass.
        A neuron is considered dead if it's saturated (> threshold) for ALL samples in the batch.
        """
        if self.last_h is None:
            return 0.0
            
        # h: (B, hidden_size)
        h = self.last_h.detach()
        saturated_mask = h.abs() > threshold # (B, hidden_size)
        
        # A neuron is "dead" if it is saturated across the whole batch
        dead_neurons = saturated_mask.all(dim=0) # (hidden_size,)
        
        return dead_neurons.float().mean().item()

    def get_activation_stats(self):
        """Compute activation statistics and histogram on the last forward pass."""
        if self.last_h is None:
            return {"mean": 0.0, "std": 0.0, "hist": []}
            
        h = self.last_h.detach().cpu()
        return {
            "mean": h.mean().item(),
            "std": h.std().item(),
            "hist": torch.histc(h, bins=20, min=-1, max=1).tolist()
        }

    def get_weight_stats(self):
        """Compute statistics for weights and biases to calculate Karpathy Ratios."""
        stats = {}
        for name, param in self.named_parameters():
            p = param.detach().cpu()
            stats[name] = {
                "mean": p.mean().item(),
                "std": p.std().item(),
            }
        return stats

    def get_embedding_quality_metrics(self, tokenizer=None):
        """Compute quantitative metrics for embedding quality."""
        W = self.C.weight.detach().cpu()
        norms = torch.norm(W, dim=1)
        
        # Pairwise distances
        dists = torch.cdist(W, W)
        avg_dist = dists.mean().item()
        
        metrics = {
            "mean_norm": norms.mean().item(),
            "std_norm": norms.std().item(),
            "avg_pairwise_dist": avg_dist,
            "max_pairwise_dist": dists.max().item()
        }

        # Objective 3: Intra-group vs inter-group (vowels vs consonants)
        if tokenizer is not None:
            vowels = set("aeiouAEIOU")
            v_idx = [i for i, c in tokenizer.itos.items() if c in vowels]
            c_idx = [i for i, c in tokenizer.itos.items() if c not in vowels and c.isalpha()]
            
            if v_idx and c_idx:
                v_embeds = W[v_idx]
                c_embeds = W[c_idx]
                
                v_centroid = v_embeds.mean(dim=0)
                c_centroid = c_embeds.mean(dim=0)
                
                intra_v = torch.cdist(v_embeds, v_embeds).mean().item()
                intra_c = torch.cdist(c_embeds, c_embeds).mean().item()
                inter_vc = torch.norm(v_centroid - c_centroid).item()
                
                metrics.update({
                    "vowel_consonant_separation": inter_vc,
                    "vowel_intra_dist": intra_v,
                    "consonant_intra_dist": intra_c
                })
        
        return metrics

    def get_gradient_flow_health(self):
        """Objective 4.3: Detect vanishing/exploding gradients."""
        layer_norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                layer_norms[name] = param.grad.norm().item()
            else:
                layer_norms[name] = 0.0
                
        total_norm = sum(v**2 for v in layer_norms.values())**0.5
        
        return {
            "total_grad_norm": total_norm,
            "layer_norms": layer_norms,
            "status": "healthy" if 0.0001 < total_norm < 10.0 else "warning"
        }

    def get_representation_dynamics(self, prev_state_dict=None):
        """Objective 4.4: Embedding drift and weight update magnitude."""
        dynamics = {}
        if prev_state_dict is None:
            return dynamics
            
        with torch.no_grad():
            curr_emb = self.C.weight.detach().cpu()
            prev_emb = prev_state_dict['C.weight'].detach().cpu()
            
            drift = torch.norm(curr_emb - prev_emb).item()
            cos_sim = F.cosine_similarity(curr_emb.view(-1), prev_emb.view(-1), dim=0).item()
            
            dynamics["embedding_drift"] = drift
            dynamics["embedding_cosine_sim"] = cos_sim
            
            # Layer-wise update magnitudes
            updates = {}
            for name, param in self.named_parameters():
                if name in prev_state_dict:
                    curr_p = param.detach().cpu()
                    prev_p = prev_state_dict[name].detach().cpu()
                    updates[name] = torch.norm(curr_p - prev_p).item()
            dynamics["update_magnitudes"] = updates
            
        return dynamics

    def generate(self, idx, max_new_tokens):
        """
        Simple generation for qualitative evaluation.
        idx is (B, T) tensor of indices in the current context.
        """
        for _ in range(max_new_tokens):
            # crop idx to the last context_size tokens
            idx_cond = idx[:, -self.context_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # focus only on the last time step
            # pluck the probabilities from the softmax
            probs = F.softmax(logits, dim=-1) # (B, V)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
        
    def get_model_stats(self):
        """Calculate total parameters and estimate inference latency."""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Estimate inference latency (ms)
        import time
        t0 = time.time()
        with torch.no_grad():
            dummy_input = torch.zeros((1, self.context_size), dtype=torch.long).to(DEVICE)
            # Warmup
            for _ in range(10):
                self(dummy_input)
            t1 = time.time()
            # Benchmark
            iters = 100
            for _ in range(iters):
                self(dummy_input)
            t2 = time.time()
            
        latency_ms = ((t2 - t1) / iters) * 1000
        
        return {
            "total_parameters": total_params,
            "inference_latency_ms": latency_ms
        }
