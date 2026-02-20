from pathlib import Path
import os
import torch

GRID = {
    "emb_dim": [2, 3, 6, 10, 16, 32],
    "hidden_size": [32, 64, 128, 256, 512, 1024],
    "learning_rate": [0.2, 0.1, 0.01]
}

def get_runs():
    runs = []
    for emb_dim in GRID["emb_dim"]:
        for hidden_size in GRID["hidden_size"]:
            for lr in GRID["learning_rate"]:
                runs.append({"emb_dim": emb_dim, "hidden_size": hidden_size, "learning_rate": lr})
    return runs

def check_worker_progress():
    runs = get_runs()
    ranges = [(0, 27), (27, 54), (54, 81), (81, 108)]
    checkpoint_dir = Path("checkpoints/mlp_grid")
    
    for i, (start, end) in enumerate(ranges):
        subset = runs[start:end]
        found = 0
        latest_idx = -1
        for j, cfg in enumerate(subset):
            fname = f"mlp_E{cfg['emb_dim']}_H{cfg['hidden_size']}_LR{cfg['learning_rate']}.pt"
            if (checkpoint_dir / fname).exists():
                found += 1
                latest_idx = j
        
        print(f"Worker {i} ({start}-{end}): {found}/{len(subset)} done.")
        if latest_idx >= 0:
            cfg = subset[latest_idx]
            print(f"  Last found: E{cfg['emb_dim']} H{cfg['hidden_size']} LR{cfg['learning_rate']} (Index {start + latest_idx})")
        else:
            print("  None found yet.")

if __name__ == "__main__":
    check_worker_progress()
