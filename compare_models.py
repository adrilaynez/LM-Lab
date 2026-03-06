"""Compare val_loss between old (v1) and new (v2) training runs."""
import json
from pathlib import Path

base = Path("checkpoints/pedagogical")

for group in ["depth_comparison", "stability_grid", "big_models", "lr_sweep",
              "scale_stability", "data_size", "dropout_experiment", "overtraining_timeline"]:
    p = base / group / "summary.json"
    if not p.exists():
        print(f"\n=== {group}: NOT FOUND ===")
        continue
    with open(p) as f:
        models = json.load(f)
    print(f"\n=== {group}: {len(models)} models ===")
    for m in sorted(models, key=lambda x: x.get("label", "")):
        lr = m.get("config", {}).get("learning_rate", "?")
        opt = m.get("config", {}).get("optimizer_type", "?")
        h = m.get("config", {}).get("hidden_size", "?")
        nl = m.get("config", {}).get("num_layers", "?")
        ctx = m.get("config", {}).get("context_size", "?")
        tl = m.get("final_train_loss", "?")
        vl = m.get("final_val_loss", "?")
        div = " DIVERGED" if m.get("diverged") else ""
        if isinstance(tl, (int, float)) and tl != float("inf"):
            tr_s = f"{tl:.4f}"
        else:
            tr_s = "INF"
        if isinstance(vl, (int, float)) and vl != float("inf"):
            vl_s = f"{vl:.4f}"
        else:
            vl_s = "INF"
        if isinstance(tl, (int, float)) and isinstance(vl, (int, float)) and tl != float("inf"):
            gap = f" gap={vl - tl:+.4f}"
        else:
            gap = ""
        label = m.get("label", "?")
        print(f"  {label:<50} H={h} L={nl} ctx={ctx} opt={opt} lr={lr} train={tr_s} val={vl_s}{gap}{div}")
