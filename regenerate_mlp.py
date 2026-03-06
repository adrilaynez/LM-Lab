#!/usr/bin/env python3
"""
Regenerate MLP checkpoint with current architecture
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

def main():
    print("🔄 Regenerating MLP checkpoint with current architecture...")
    
    # Setup
    torch.manual_seed(config.SEED)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Load data
    print(f"📚 Loading data from {config.DATA_PATH}...")
    raw_text = load_data(config.DATA_PATH)
    
    tokenizer = CharTokenizer()
    tokenizer.train(raw_text)
    vocab_size = tokenizer.vocab_size
    
    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Create MLP model with current architecture
    print("🧠 Creating MLP model with current architecture...")
    ModelClass = get_model_class('mlp')
    model = ModelClass(
        vocab_size=vocab_size,
        context_size=config.BLOCK_SIZE,
        emb_dim=10,
        hidden_size=200,
        num_layers=1,
        seed=config.SEED
    )
    model = model.to(config.DEVICE)
    
    # Print model structure
    print("\n📋 Model structure:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Vocabulary size: {vocab_size}")
    print(f"   - Training data size: {len(train_data):,} tokens")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Quick training loop (just 1000 steps to create a working checkpoint)
    print("\n🚀 Quick training (1000 steps)...")
    loss_history = []
    
    for step in range(1000):
        xb, yb = get_batch(train_data, config.BATCH_SIZE, config.BLOCK_SIZE, config.DEVICE)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")
            loss_history.append(loss.item())
    
    # Save checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'mlp.pt')
    print(f"\n💾 Saving checkpoint to {checkpoint_path}...")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'context_size': config.BLOCK_SIZE,
            'emb_dim': 10,
            'hidden_size': 200,
            'num_layers': 1,
            'seed': config.SEED
        },
        'loss_history': loss_history,
        'step': 1000
    }
    
    torch.save(checkpoint, checkpoint_path)
    print("✅ Checkpoint saved successfully!")
    
    # Test loading
    print("\n🧪 Testing checkpoint loading...")
    test_checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    # Create new model and load state
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
    print("✅ Checkpoint loads successfully!")
    
    # Test generation
    print("\n🎯 Testing generation...")
    test_model.eval()
    with torch.no_grad():
        # Create a simple batch with proper context size
        batch_size = 1
        context_size = test_checkpoint['config']['context_size']
        context = torch.randint(0, test_checkpoint['config']['vocab_size'], (batch_size, context_size), device=config.DEVICE)
        logits, _ = test_model(context)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_char_idx = torch.multinomial(probs, num_samples=1)[0].item()
        print(f"  Context shape: {context.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Generated next char index: {next_char_idx}")
    
    print("\n🎉 MLP checkpoint regeneration complete!")

if __name__ == "__main__":
    main()
