#!/usr/bin/env python3
"""
Create MLP grid checkpoints with proper snapshot structure
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

def create_checkpoint_with_snapshots(emb_dim, hidden_size, lr, steps=1000):
    """Create a single MLP checkpoint with snapshot structure"""
    
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
    
    # Create checkpoint structure with snapshots
    checkpoint = {
        'config': {
            'vocab_size': vocab_size,
            'context_size': config.BLOCK_SIZE,
            'emb_dim': emb_dim,
            'hidden_size': hidden_size,
            'num_layers': 1,
            'seed': config.SEED,
            'learning_rate': lr
        },
        'metadata': {
            'vocab': tokenizer.chars,
            'vocab_size': vocab_size,
            'training_data': config.DATA_PATH,
            'model_type': 'mlp'
        },
        'snapshots': {},
        'metrics_log': {}
    }
    
    # Training loop with snapshots
    snapshot_intervals = [0, steps//2, steps]
    
    for step in range(steps + 1):
        xb, yb = get_batch(train_data, config.BATCH_SIZE, config.BLOCK_SIZE, config.DEVICE)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Save snapshot at intervals
        if step in snapshot_intervals:
            snapshot_key = f"step_{step}"
            
            # Get model internals for interpretability
            model.eval()
            with torch.no_grad():
                # Get embedding matrix
                emb_matrix = model.C.weight.detach().cpu() if model.C is not None else torch.eye(vocab_size)
                
                # Get activations on a sample batch
                sample_xb, _ = get_batch(train_data, min(32, len(train_data)), config.BLOCK_SIZE, config.DEVICE)
                _ = model(sample_xb)  # Forward pass to populate internals
                h_pre = model.last_h_pre.detach().cpu() if model.last_h_pre is not None else None
                h = model.last_h.detach().cpu() if model.last_h is not None else None
            
            checkpoint['snapshots'][snapshot_key] = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'loss': loss.item(),
                'metrics': {
                    'train_loss': loss.item(),
                    'val_loss': loss.item()  # Same for now
                },
                'interpretability': {
                    'embedding_matrix': emb_matrix.tolist(),
                    'hidden_preactivations': h_pre.tolist() if h_pre is not None else None,
                    'hidden_activations': h.tolist() if h is not None else None
                }
            }
            
            checkpoint['metrics_log'][snapshot_key] = {
                'train_loss': {'value': loss.item(), 'step': step},
                'val_loss': {'value': loss.item(), 'step': step}
            }
            
            print(f"    Snapshot {step}: loss = {loss.item():.4f}")
        
        model.train()
    
    # Save checkpoint
    checkpoint_name = f"mlp_E{emb_dim}_H{hidden_size}_LR{lr}.pt"
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'mlp_grid', checkpoint_name)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"  ✅ Saved: {checkpoint_name}")
    
    return checkpoint_path

def main():
    print("🔄 Creating MLP grid checkpoints with snapshot structure...")
    
    # Ensure directory exists
    os.makedirs(os.path.join(config.CHECKPOINT_DIR, 'mlp_grid'), exist_ok=True)
    
    # Define the configurations we need
    configs = [
        (2, 4, 0.1),    # The one the frontend is trying to use
        (2, 32, 0.1),   # Small but exists
        (10, 128, 0.1), # Medium
        (16, 256, 0.1), # Larger
    ]
    
    print(f"\n🎯 Creating {len(configs)} checkpoints with snapshots...")
    
    for emb_dim, hidden_size, lr in configs:
        print(f"\n📦 Creating checkpoint: E{emb_dim}_H{hidden_size}_LR{lr}")
        try:
            create_checkpoint_with_snapshots(emb_dim, hidden_size, lr)
        except Exception as e:
            print(f"  ❌ Error creating checkpoint: {e}")
    
    print("\n🎉 MLP grid checkpoint creation complete!")
    
    # Test one of them
    print("\n🧪 Testing a checkpoint...")
    test_emb, test_hidden, test_lr = configs[0]
    checkpoint_name = f"mlp_E{test_emb}_H{test_hidden}_LR{test_lr}.pt"
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'mlp_grid', checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Test checkpoint exists: {checkpoint_name}")
        
        # Test loading
        test_checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        print(f"✅ Has snapshots: {len(test_checkpoint.get('snapshots', {}))} snapshots")
        print(f"✅ Has metadata: {'metadata' in test_checkpoint}")
        print(f"✅ Has metrics_log: {'metrics_log' in test_checkpoint}")
    else:
        print(f"❌ Test checkpoint not found: {checkpoint_name}")

if __name__ == "__main__":
    main()
