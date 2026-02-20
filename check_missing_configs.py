from pathlib import Path

GRID = {
    "emb_dim": [2, 3, 6, 10, 16, 32],
    "hidden_size": [32, 64, 128, 256, 512, 1024],
    "learning_rate": [0.2, 0.1, 0.01]
}

CHECKPOINT_DIR = Path("checkpoints/mlp_grid")

def check_missing():
    total = 0
    missing = []
    found = []
    for emb in GRID["emb_dim"]:
        for hidden in GRID["hidden_size"]:
            for lr in GRID["learning_rate"]:
                total += 1
                fname = f"mlp_E{emb}_H{hidden}_LR{lr}.pt"
                if not (CHECKPOINT_DIR / fname).exists():
                    missing.append(fname)
                else:
                    found.append(fname)
    
    print(f"Total Expected: {total}")
    print(f"Found: {len(found)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print("\nFirst 10 missing:")
        for m in missing[:10]:
            print(m)

if __name__ == "__main__":
    check_missing()
