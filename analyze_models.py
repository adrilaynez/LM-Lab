"""Analyze all MLP grid models and output a comprehensive summary."""
import torch, json
from pathlib import Path
from collections import defaultdict

GRID_DIR = Path("checkpoints/mlp_grid")
TIMELAPSE_DIR = Path("checkpoints/mlp_timelapse")

results = []
for d, src in [(GRID_DIR, "grid"), (TIMELAPSE_DIR, "timelapse")]:
    for f in sorted(d.glob("*.pt")):
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            cfg = ck.get("config", {})
            snaps = ck.get("snapshots", {})
            sorted_steps = sorted(snaps.keys(), key=lambda s: int(s.split("_")[1])) if snaps else []
            last = snaps[sorted_steps[-1]].get("metrics", {}) if sorted_steps else {}
            tr = last.get("train_loss", cfg.get("final_loss"))
            vl = last.get("val_loss")
            results.append({
                "f": f.stem, "src": src,
                "e": cfg.get("embedding_dim", cfg.get("emb_dim", 0)),
                "h": cfg.get("hidden_size", 0),
                "lr": cfg.get("learning_rate", 0),
                "nl": cfg.get("num_layers", 1),
                "dr": cfg.get("dropout", 0.0),
                "p": cfg.get("total_parameters", 0),
                "tr": round(tr, 4) if tr else None,
                "vl": round(vl, 4) if vl else None,
            })
        except Exception as ex:
            print("ERR", f.name, ex)

# Write as JSON for easy reading
with open("analysis_results.json", "w") as fp:
    json.dump(results, fp, indent=2)

print("Total models:", len(results))
print("Grid:", len([r for r in results if r["src"]=="grid"]))
print("Timelapse:", len([r for r in results if r["src"]=="timelapse"]))
print("Dropout models:", len([r for r in results if r["dr"]>0]))
print("Saved to analysis_results.json")
