"""
Parallel wrapper for MLP precomputation.
Accepts start and end indices for the grid.
"""
import sys
import os
from pathlib import Path

# Fix import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models.mlp_precompute as mlp_main
import torch

def main():
    if len(sys.argv) < 3:
        print("Usage: python mlp_precompute_parallel.py <start_idx> <end_idx>")
        return
    
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    
    print(f"🚀 Starting Parallel Slice: {start_idx} to {end_idx}")
    
    # Setup data
    raw_text = mlp_main.load_data(mlp_main.DATA_PATH)
    tokenizer = mlp_main.CharTokenizer()
    tokenizer.train(raw_text)
    
    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    runs = mlp_main.get_runs()
    subset = runs[start_idx:end_idx]
    
    registry = []
    
    for i, run_cfg in enumerate(subset):
        global_idx = start_idx + i
        print(f"\n[{global_idx+1}/{len(runs)}] Parallel Worker: {start_idx}-{end_idx} | Config: E{run_cfg['emb_dim']} H{run_cfg['hidden_size']} LR{run_cfg['learning_rate']}")
        
        fname = f"mlp_E{run_cfg['emb_dim']}_H{run_cfg['hidden_size']}_LR{run_cfg['learning_rate']}.pt"
        save_path = mlp_main.OUTPUT_DIR / fname
        
        if save_path.exists():
            print(f"   ⏩ Skipping, already exists.")
            continue

        try:
            results = mlp_main.train_run(run_cfg, train_data, val_data, tokenizer)
            torch.save(results, save_path)
            
            print(f"   ✅ Saved (Score: {results['metadata']['score']:.2f})")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🎉 Parallel Worker {start_idx}-{end_idx} Complete!")

if __name__ == "__main__":
    main()
