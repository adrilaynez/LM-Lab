#!/usr/bin/env python3
"""
Regenerate MLP grid checkpoints with current architecture
"""

import torch
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import get_model_class
from utils.tokenizer import CharTokenizer
from utils.data import load_data, get_batch
import config

def create_checkpoint(emb_dim, hidden_size, lr, steps=1000):
    """Create a single MLP checkpoint with given parameters"""
    
    # Setup
    torch.manual_seed(config.SEED)
    
    # Load data
    raw_text = load_data(config.DATA_PATH)
    tokenizer = CharTokenizer()
    tokenizer.train(raw_text)
    vocab_size = tokenizer.vocab_size
    
    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    
    # Create MLP model
    ModelClass = get_model_class('mlp')
    model = ModelClass(
        vocab_size=vocab_size,
        context_size=config.BLOCK_SIZE,
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        num_layers=1,
        seed=config.SEED
    )
    model = model.to(config.DEVICE)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Quick training loop
    for step in range(steps):
        xb, yb = get_batch(train_data, config.BATCH_SIZE, config.BLOCK_SIZE, config.DEVICE)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"    E{emb_dim}_H{hidden_size}_LR{lr} - Step {step}: loss = {loss.item():.4f}")
    
    # Save checkpoint
    checkpoint_name = f"mlp_E{emb_dim}_H{hidden_size}_LR{lr}.pt"
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'mlp_grid', checkpoint_name)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'context_size': config.BLOCK_SIZE,
            'emb_dim': emb_dim,
            'hidden_size': hidden_size,
            'num_layers': 1,
            'seed': config.SEED,
            'learning_rate': lr
        },
        'step': steps
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"  ✅ Saved: {checkpoint_name}")
    
    return checkpoint_path

def main():
    print("🔄 Regenerating MLP grid checkpoints with current architecture...")
    
    # Ensure directory exists
    os.makedirs(os.path.join(config.CHECKPOINT_DIR, 'mlp_grid'), exist_ok=True)
    
    # Define the configurations we need (based on what the frontend uses)
    configs = [
        # Common configurations used by the frontend
        (2, 4, 0.1),    # The one the frontend is trying to use
        (2, 32, 0.1),   # Small but exists
        (10, 128, 0.1), # Medium
        (16, 256, 0.1), # Larger
    ]
    
    print(f"\n🎯 Creating {len(configs)} checkpoints...")
    
    for emb_dim, hidden_size, lr in configs:
        print(f"\n📦 Creating checkpoint: E{emb_dim}_H{hidden_size}_LR{lr}")
        try:
            create_checkpoint(emb_dim, hidden_size, lr)
        except Exception as e:
            print(f"  ❌ Error creating checkpoint: {e}")
    
    print("\n🎉 MLP grid checkpoint regeneration complete!")
    
    # Test one of them
    print("\n🧪 Testing a checkpoint...")
    test_emb, test_hidden, test_lr = configs[0]
    checkpoint_name = f"mlp_E{test_emb}_H{test_hidden}_LR{test_lr}.pt"
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'mlp_grid', checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Test checkpoint exists: {checkpoint_name}")
        
        # Test loading
        test_checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        ModelClass = get_model_class('mlp')
        test_model = ModelClass(
            vocab_size=test_checkpoint['config']['vocab_size'],
            context_size=test_checkpoint['config']['context_size'],
            emb_dim=test_checkpoint['config']['emb_dim'],
            hidden_size=test_checkpoint['config']['hidden_size'],
            num_layers=test_checkpoint['config']['num_layers'],
            seed=test_checkpoint['config']['seed']
        )
        test_model = test_model.to(config.DEVICE)
        test_model.load_state_dict(test_checkpoint['model_state_dict'])
        print("✅ Test checkpoint loads successfully!")
    else:
        print(f"❌ Test checkpoint not found: {checkpoint_name}")

if __name__ == "__main__":
    main()
