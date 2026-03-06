import json
import sys
from pathlib import Path

# Force UTF-8 encoding for Windows standard output
sys.stdout.reconfigure(encoding='utf-8')

def print_summary(group_dir):
    p = Path(rf"c:\Projects\LM-Lab\checkpoints\pedagogical\{group_dir}\summary.json")
    if not p.exists():
        print(f"Directory or summary not found for {group_dir}")
        return
        
    with open(p, 'r') as f:
        data = json.load(f)
        
    print(f"\n{'='*50}")
    print(f"=== {group_dir.upper()} ===")
    print(f"{'='*50}")
    
    # Sort differently based on the group
    if group_dir == "lr_sweep":
        data.sort(key=lambda x: x["config"].get("learning_rate", 0))
    elif group_dir == "dropout_experiment":
        data.sort(key=lambda x: x["config"].get("dropout_rate", 0))
        
    for item in data:
        cfg = item["config"]
        # Distinguish interesting configs
        lr = cfg.get("learning_rate", cfg.get("lr", "N/A"))
        dp = cfg.get("dropout_rate", "N/A")
        train_loss = item.get("final_train_loss", "N/A")
        val_loss = item.get("final_val_loss", "N/A")
        
        # calculate generalization gap if both are floats
        gap = "N/A"
        if isinstance(train_loss, float) and isinstance(val_loss, float):
            gap = f"{(val_loss - train_loss):.4f}"
            
        print(f"Model: {item['label']:20} | LR: {lr:<6} | Dropout: {dp:<4} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Gap: {gap}")
        
    # For overtraining timeline we want to see the loss curve milestones
    if group_dir == "overtraining_timeline" and len(data) > 0:
        print("\n--- Milestone Snapshots ---")
        snapshots = data[0].get("text_snapshots", [])
        for snap in snapshots:
            print(f"Step {snap['step']:>6} | Text: {snap['text']}\n")


if __name__ == "__main__":
    groups = ["lr_sweep", "dropout_experiment", "overtraining_timeline", "weight_tying", "weight_tying_graham"]
    for g in groups:
        print_summary(g)
