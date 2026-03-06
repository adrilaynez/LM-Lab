import argparse
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
_ROOT = Path(r"c:\Projects\LM-Lab")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import train_pedagogical_v2
# OUTPUT_ROOT is defined inside train_pedagogical_v2, not api.config
OUTPUT_ROOT = train_pedagogical_v2.OUTPUT_ROOT

def train_depth_new_comparison(max_workers=4):
    target_dir = OUTPUT_ROOT / "depth_new_comparison"
    layers_to_train = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20]
    
    train_pedagogical_v2.tprint("\n📚 Loading data...")
    train_data, val_data, tokenizer = train_pedagogical_v2.load_default_data()
    
    tasks = []
    FIXED_SEED = 424242 
    
    for i, n_layers in enumerate(layers_to_train):
        label = f"depth_L{n_layers}"
        tasks.append({
            "label": label,
            "fname": f"{label}.pt",
            "kwargs": {
                "vocab_size": tokenizer.vocab_size,
                "context_size": 4, 
                "emb_dim": 10,     
                "hidden_size": 128,
                "num_layers": n_layers,
                "init_strategy": "random", 
                "use_batchnorm": False,
                "use_residual": False,
                "max_steps": 80_000, 
                "lr": 0.01, # Increased LR
                "train_data": train_data,
                "val_data": val_data,
                "tokenizer": tokenizer,
                "label": label,
                "grad_clip_val": None,
                "model_seed": FIXED_SEED, # Identical Seed
            },
        })
        
    train_pedagogical_v2._run_concurrent(
        tasks, target_dir,
        skip_existing=False, 
        max_workers=max_workers,
        group_name="GROUP NEW — depth_new_comparison (SGD lr=0.01, random init, SAME SEED)",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    train_depth_new_comparison(max_workers=args.workers)
