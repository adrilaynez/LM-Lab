import torch
import os
import glob

files = glob.glob('c:/Projects/LM-Lab/checkpoints/mlp_timelapse/*.pt')

for f in files:
    ckpt = torch.load(f, map_location='cpu')
    keys = list(ckpt['snapshots'].keys())
    if len(keys) > 0 and isinstance(keys[0], int):
        print(f"Fixing {f}...")
        new_snapshots = {}
        for step, emb_matrix in ckpt['snapshots'].items():
            new_key = f"step_{step}"
            
            # The API extracts emb_matrix by computing:
            # emb_matrix = interp.get("embedding_matrix", [])
            new_snapshots[new_key] = {
                "interpretability": {
                    "embedding_matrix": emb_matrix
                }
            }
            
            # inference.py expects model_state_dict inside the last snapshot
            if step == 50000:
                new_snapshots[new_key]["model_state_dict"] = ckpt.get("model_state_dict", {})
                
        ckpt['snapshots'] = new_snapshots
        torch.save(ckpt, f)
        print(f"  Fixed!")
    else:
        print(f"Skipping {f}, already formatted correctly? Keys: {keys[:2]}")
