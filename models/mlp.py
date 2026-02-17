"""
Multi-Layer Perceptron (MLP) Language Model
Character embeddings + 2 hidden layers + output layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import LMEngine


class MLPModel(LMEngine):
    """
    MLP Language Model with character embeddings.
    Architecture:
    - Input: character indices (batch_size, block_size)
    - Embedding: (batch_size, block_size) -> (batch_size, block_size, embedding_dim)
    - Flatten: (batch_size, block_size * embedding_dim)
    - Hidden1: -> (batch_size, hidden_dim1)
    - Hidden2: -> (batch_size, hidden_dim2)
    - Output: -> (batch_size, vocab_size)
    """
    
    def __init__(self, vocab_size, embedding_dim=10, hidden_dim1=64, hidden_dim2=32, block_size=8):
        super().__init__(vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.block_size = block_size
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # First hidden layer
        input_size = block_size * embedding_dim
        self.hidden1 = nn.Linear(input_size, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        
        # Second hidden layer
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        
        # Output layer
        self.output = nn.Linear(hidden_dim2, vocab_size)
        
        # Activation function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, idx, targets=None):
        """
        Forward pass through MLP.
        idx shape: (batch_size, block_size)
        targets shape: (batch_size, block_size) or None
        """
        B, T = idx.shape
        
        # Embedding layer
        x = self.embedding(idx)  # (B, T, embedding_dim)
        
        # Store for activation visualization
        self.embedding_output = x.detach()
        
        # Flatten for fully connected layers
        x = x.reshape(B, -1)  # (B, T * embedding_dim)
        
        # First hidden layer with batch norm and activation
        x = self.hidden1(x)  # (B, hidden_dim1)
        self.hidden1_pre_activation = x.detach()
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        self.hidden1_activation = x.detach()
        
        # Second hidden layer with batch norm and activation
        x = self.hidden2(x)  # (B, hidden_dim2)
        self.hidden2_pre_activation = x.detach()
        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        self.hidden2_activation = x.detach()
        
        # Output layer
        logits = self.output(x)  # (B, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten targets to match batch dimension
            targets_flat = targets.reshape(-1)
            logits_flat = logits
            
            # Handle the case where we need to predict at each position
            # For MLP we process entire sequence at once, so logits are per sequence
            # But targets are per token - need to handle this properly
            if logits.shape[0] != targets_flat.shape[0]:
                # Reshape logits to match targets: replicate logits for each token in sequence
                logits_flat = logits.repeat_interleave(T, dim=0)
                
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def get_internals(self, idx, targets=None):
        """
        Return internal representations for visualization.
        """
        with torch.no_grad():
            _, _ = self.forward(idx, targets)
            
        return {
            "embedding_output": self.embedding_output,
            "hidden1_pre_activation": self.hidden1_pre_activation,
            "hidden1_activation": self.hidden1_activation,
            "hidden2_pre_activation": self.hidden2_pre_activation,
            "hidden2_activation": self.hidden2_activation,
            "hidden1_weights": self.hidden1.weight.detach(),
            "hidden2_weights": self.hidden2.weight.detach(),
            "output_weights": self.output.weight.detach(),
        }
