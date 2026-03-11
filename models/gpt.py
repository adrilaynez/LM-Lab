"""
GPT — Character-level Generative Pre-trained Transformer
---------------------------------------------------------
Configurable decoder-only Transformer for character-level language modeling.
Architecture: Embedding + PosEncoding → N × (LN → MHA → Res → LN → FFN → Res) → LN → Linear Head

Supports:
- Configurable depth (n_blocks), width (d_model), heads (n_heads), FFN ratio
- Causal masking (built-in)
- Attention pattern extraction for visualization
- Per-layer hidden state extraction
- Temperature-controlled generation with top-k sampling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask — lower triangular
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        # Store attention weights for visualization (set during forward)
        self.last_attn_weights = None

    def forward(self, x, store_attention: bool = False):
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, C)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        if store_attention:
            self.last_attn_weights = att.detach().cpu()

        att = self.attn_dropout(att)

        # Weighted sum of values
        out = att @ v  # (B, nh, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → MHA → Res → LN → FFN → Res."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, store_attention: bool = False):
        x = x + self.attn(self.ln1(x), store_attention=store_attention)
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """
    Character-level GPT.

    Args:
        vocab_size: Number of unique characters
        d_model: Embedding / hidden dimension (default 128)
        n_heads: Number of attention heads (default 4)
        n_blocks: Number of Transformer blocks (default 4)
        d_ff: FFN inner dimension (default 4 * d_model)
        block_size: Maximum context length (default 256)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        d_ff: int = None,
        block_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.block_size = block_size
        self.dropout_rate = dropout
        self.model_type = "gpt"

        # Token + positional embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, self.d_ff, block_size, dropout)
            for _ in range(n_blocks)
        ])

        # Final layer norm + linear head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding and output head weights
        self.head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, store_attention: bool = False):
        """
        Forward pass.

        Args:
            idx: (B, T) integer token indices
            targets: (B, T) integer target indices (optional, for loss)
            store_attention: if True, store attention weights in each block

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar (if targets provided)
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Embeddings
        tok = self.tok_emb(idx)  # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
        x = self.emb_dropout(tok + pos)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, store_attention=store_attention)

        # Final norm + head
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressive generation.

        Args:
            idx: (B, T) starting token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top-k tokens
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, V)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_with_probs(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Generate tokens AND return per-step probability distributions.
        Returns: (generated_idx, list_of_prob_dicts)
        """
        all_probs = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs[0].cpu().tolist())
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx, all_probs

    @torch.no_grad()
    def get_attention_maps(self, idx):
        """
        Run forward pass storing attention and return per-layer attention maps.
        Returns: list of (n_heads, T, T) tensors, one per block.
        """
        self(idx, store_attention=True)
        maps = []
        for block in self.blocks:
            if block.attn.last_attn_weights is not None:
                maps.append(block.attn.last_attn_weights[0])  # (n_heads, T, T)
        return maps

    @torch.no_grad()
    def get_hidden_states(self, idx):
        """
        Run forward pass and capture hidden states after each block.
        Returns: list of (T, d_model) tensors, one per block + initial embedding.
        """
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        states = [x[0].cpu()]  # Initial embedding state
        for block in self.blocks:
            x = block(x)
            states.append(x[0].cpu())  # After each block

        return states

    def get_config_dict(self):
        """Return model configuration as a serializable dict."""
        return {
            "model_type": "gpt",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_blocks": self.n_blocks,
            "d_ff": self.d_ff,
            "block_size": self.block_size,
            "dropout": self.dropout_rate,
        }

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
