#!/usr/bin/env python3
"""
Parallel training script for pedagogical MLP models.
Optimized for limited GPU memory by training models sequentially 
but using multiple processes for efficiency.
"""

import os
import sys
import json
import time
import torch
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from utils.tokenizer import CharTokenizer
from models.mlp_deep import MLPDeepModel

# Configuration
DATA_PATH = PROJECT_ROOT / "data" / "tinyshakespeare_clean.txt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "pedagogical" / "stability_grid"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
BATCH_SIZE = 64
EPOCHS = 80000
LEARNING_RATE = 0.001

# Global lock for GPU access
gpu_lock = Lock()

def load_data():
    """Load and prepare training data."""
    print(f"📚 Loading data from {DATA_PATH} ...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    
    # Split data
    n = int(0.9 * len(text))
    train_text = text[:n]
    val_text = text[n:]
    
    # Encode
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    
    print(f"    {len(text):,} chars | vocab={tokenizer.vocab_size} | train={len(train_data):,} val={len(val_data):,}")
    return train_data, val_data, tokenizer

def get_batch(data, i, batch_size, context_size):
    """Get a batch of data."""
    # Create sequences of context_size + 1
    seq_len = context_size + 1
    batch = data[i:i+batch_size*seq_len].view(batch_size, seq_len)
    x = batch[:, :context_size]
    y = batch[:, 1:context_size+1]
    return x, y

def evaluate_model(model, data, batch_size, context_size, device):
    """Evaluate model on data."""
    model.eval()
    total_loss = 0
    num_batches = 100  # Use fixed number of batches for evaluation
    
    with torch.no_grad():
        for _ in range(num_batches):
            ix = torch.randint(len(data) - context_size, (batch_size,))
            x = torch.stack([data[i:i+context_size] for i in ix])
            y = torch.stack([data[i+1:i+1+context_size] for i in ix])
            x, y = x.to(device), y.to(device)
            
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    
    return total_loss / num_batches

def train_single_model(config, train_data, val_data, tokenizer, model_id):
    """Train a single model configuration."""
    # Acquire GPU lock
    with gpu_lock:
        print(f"\n🚀 Starting {model_id} | {config['num_layers'] * config['hidden_size']:,} params | device={DEVICE}")
        
        # Set seed
        torch.manual_seed(SEED)
        
        # Create model
        model = MLPDeepModel(
            vocab_size=tokenizer.vocab_size,
            context_size=config['context_size'],
            emb_dim=config['emb_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            init_strategy=config['init_strategy'],
            use_batchnorm=config['use_batchnorm'],
            use_residual=config['use_residual'],
            seed=SEED
        ).to(DEVICE)
        
        # Optimizer with cosine annealing
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        # Training loop
        model.train()
        start_time = time.time()
        
        for step in range(EPOCHS):
            # Get batch - use original working approach
            ix = torch.randint(len(train_data) - config['context_size'], (BATCH_SIZE,))
            x = torch.stack([train_data[i:i+config['context_size']] for i in ix])
            y = torch.stack([train_data[i+1:i+1+config['context_size']] for i in ix])
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Progress reporting
            if step % 10000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                train_loss = loss.item()
                
                # Evaluate on validation set
                val_loss = evaluate_model(model, val_data, BATCH_SIZE, config['context_size'], DEVICE)
                
                print(f"    step {step:6d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}")
        
        # Final evaluation
        final_train_loss = evaluate_model(model, train_data, BATCH_SIZE, config['context_size'], DEVICE)
        final_val_loss = evaluate_model(model, val_data, BATCH_SIZE, config['context_size'], DEVICE)
        
        training_time = time.time() - start_time
        
        print(f"    ✓ done  train={final_train_loss:.4f}  val={final_val_loss:.4f}  time={training_time:.1f}s")
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'training_time': training_time,
            'model_id': model_id
        }
        
        checkpoint_path = CHECKPOINT_DIR / f"{model_id}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved → {model_id}.pt")
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
        
        return {
            'model_id': model_id,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'training_time': training_time,
            'config': config
        }

def get_group2_configs():
    """Get all Group 2 configurations."""
    depths = [1, 2, 3, 4, 6]
    techniques = [
        ('none', False, False, 'random'),
        ('kaiming', False, False, 'kaiming'),
        ('kaiming+BN', True, False, 'kaiming'),
        ('kaiming+BN+residual', True, True, 'kaiming')
    ]
    
    configs = []
    for depth in depths:
        for name, use_bn, use_res, init_strat in techniques:
            config = {
                'context_size': 3,
                'emb_dim': 10,
                'hidden_size': 128,
                'num_layers': depth,
                'init_strategy': init_strat,
                'use_batchnorm': use_bn,
                'use_residual': use_res
            }
            model_id = f"L{depth}_{name}"
            configs.append((model_id, config))
    
    return configs

def check_existing_models():
    """Check which models already exist."""
    existing = set()
    if CHECKPOINT_DIR.exists():
        for file in CHECKPOINT_DIR.glob("*.pt"):
            existing.add(file.stem)
    return existing

def train_group2_parallel(train_data, val_data, tokenizer, max_workers=2):
    """Train Group 2 models with limited parallelism for GPU memory constraints."""
    configs = get_group2_configs()
    existing = check_existing_models()
    
    # Filter out existing models
    remaining = [(mid, cfg) for mid, cfg in configs if mid not in existing]
    
    print(f"\n{'='*60}")
    print(f"GROUP 2 — Stability Technique Grid (Parallel)")
    print(f"{'='*60}")
    print(f"  Models to train: {len(remaining)}/{len(configs)}")
    print(f"  Max parallel workers: {max_workers} (GPU memory constraint)")
    print(f"  Device: {DEVICE}")
    
    if not remaining:
        print("  ✅ All models already exist!")
        return
    
    # Train models with limited parallelism
    results = []
    
    # Use ThreadPoolExecutor for I/O bound tasks and GPU lock coordination
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(train_single_model, cfg, train_data, val_data, tokenizer, mid): mid
            for mid, cfg in remaining
        }
        
        # Process completed tasks
        for future in as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
                print(f"  ✅ {model_id} completed")
            except Exception as e:
                print(f"  ❌ {model_id} failed: {e}")
    
    # Save summary
    summary_path = CHECKPOINT_DIR / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Group 2 training complete!")
    print(f"   Results saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Train pedagogical MLP models in parallel")
    parser.add_argument('--group', type=int, default=2, help='Group to train (2=stability grid)')
    parser.add_argument('--workers', type=int, default=2, help='Max parallel workers (GPU memory constraint)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=80000, help='Training epochs')
    
    args = parser.parse_args()
    
    # Update global config
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_data, val_data, tokenizer = load_data()
    
    # Train specified group
    if args.group == 2:
        train_group2_parallel(train_data, val_data, tokenizer, args.workers)
    else:
        print(f"Group {args.group} not supported yet")

if __name__ == "__main__":
    main()
