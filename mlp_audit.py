import os
import torch
import json
import numpy as np
from pathlib import Path
from tabulate import tabulate

# Configuration mirroring models/mlp_precompute.py
GRID = {
    "emb_dim": [2, 3, 6, 10, 16, 32],
    "hidden_size": [32, 64, 128, 256, 512, 1024],
    "learning_rate": [0.2, 0.1, 0.01]
}
SNAPSHOT_STEPS = [1000, 5000, 10000, 20000, 50000]
CHECKPOINT_DIR = Path("checkpoints/mlp_grid")
MANIFEST_PATH = CHECKPOINT_DIR / "manifest.json"

def audit():
    print(f"🔍 Starting Enhanced MLP Experimental Grid Audit...")
    print(f"Target Directory: {CHECKPOINT_DIR}\n")
    
    if not CHECKPOINT_DIR.exists():
        print(f"❌ Error: Checkpoint directory {CHECKPOINT_DIR} does not exist.")
        return

    results = []
    leaderboard_data = []
    total_configs = len(GRID["emb_dim"]) * len(GRID["hidden_size"]) * len(GRID["learning_rate"])
    found_count = 0
    
    for emb in GRID["emb_dim"]:
        for hidden in GRID["hidden_size"]:
            for lr in GRID["learning_rate"]:
                config_str = f"E{emb}_H{hidden}_LR{lr}"
                fname = f"mlp_{config_str}.pt"
                fpath = CHECKPOINT_DIR / fname
                
                status = "MISSING"
                snaps_present = []
                missing_data = []
                final_val_loss = None
                score = 0
                
                if fpath.exists():
                    found_count += 1
                    try:
                        data = torch.load(fpath, map_location="cpu")
                        snaps = data.get("snapshots", {})
                        snaps_present = [s for s in ["step_0"] + [f"step_{step}" for step in SNAPSHOT_STEPS] if s in snaps]
                        
                        expected_snaps = ["step_0"] + [f"step_{step}" for step in SNAPSHOT_STEPS]
                        missing_snaps = [s for s in expected_snaps if s not in snaps]
                        
                        if missing_snaps:
                            missing_data.append(f"Missing snaps: {missing_snaps}")
                        
                        # Metadata check
                        metadata = data.get("metadata", {})
                        score = metadata.get("score", 0)
                        
                        # Consistency checks
                        anomalies = []
                        losses = []
                        for s_name in expected_snaps:
                            if s_name not in snaps:
                                continue
                            snap = snaps[s_name]
                            metrics = snap.get("metrics", {})
                            val_loss = metrics.get("val_loss", 0)
                            final_val_loss = val_loss
                            losses.append(val_loss)
                            
                            if np.isnan(val_loss):
                                anomalies.append(f"{s_name} NaN Loss")
                            
                            # Dead neurons check
                            dead_p = metrics.get("dead_neurons", 0)
                            if dead_p > 0.9:
                                anomalies.append(f"{s_name} >90% Dead")
                        
                        # Check for validation spikes (Loss stability)
                        if len(losses) > 2:
                            # If final loss is significantly higher than min loss found
                            if val_loss > min(losses) * 1.5:
                                anomalies.append("Validation Spike")
                                
                        if anomalies:
                            status = f"WARNING: {', '.join(anomalies)}"
                        elif not missing_snaps:
                            status = "VALID"
                            leaderboard_data.append({
                                "config": config_str,
                                "score": score,
                                "val_loss": val_loss,
                                "dead_neurons": snaps[f"step_{SNAPSHOT_STEPS[-1]}"]["metrics"].get("dead_neurons", 0)
                            })
                        else:
                            status = "INCOMPLETE"
                            
                    except Exception as e:
                        status = "FAILED: Corrupt"
                        missing_data.append(str(e))
                
                results.append([config_str, f"{len(snaps_present)}/6", ", ".join(missing_data) if missing_data else "None", f"{final_val_loss:.4f}" if final_val_loss else "-", status])

    # Print Live Audit Table
    headers = ["Config", "Snapshots", "Issues", "Final Val Loss", "Status"]
    # Only show non-missing or interesting ones to save space
    active_results = [r for r in results if r[4] != "MISSING"]
    print(tabulate(active_results, headers=headers, tablefmt="grid"))
    
    # Objective 4: Intermediate Ranking / Leaderboard
    if leaderboard_data:
        print(f"\n🏆 PRELIMINARY LEADERBOARD (Top 10)")
        leaderboard_data.sort(key=lambda x: x["score"], reverse=True)
        lb_table = [[i+1, d["config"], f"{d['score']:.2f}", f"{d['val_loss']:.4f}", f"{d['dead_neurons']*100:.1f}%"] for i, d in enumerate(leaderboard_data[:10])]
        print(tabulate(lb_table, headers=["Rank", "Config", "Score", "Val Loss", "Dead Neurons"], tablefmt="simple"))

    print(f"\n📊 Execution Summary:")
    print(f"Total Expected: {total_configs}")
    print(f"Total Found: {found_count}")
    print(f"VALID: {sum(1 for r in results if r[4] == 'VALID')}")
    print(f"WARNING: {sum(1 for r in results if 'WARNING' in r[4])}")
    print(f"INCOMPLETE: {sum(1 for r in results if r[4] == 'INCOMPLETE')}")
    print(f"FAILED: {sum(1 for r in results if 'FAILED' in r[4])}")
    print(f"MISSING: {total_configs - found_count}")

if __name__ == "__main__":
    audit()
