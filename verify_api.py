"""Verify pedagogical API endpoints serve correct data."""
from api.services.inference import (
    get_depth_comparison, get_stability_grid,
    get_big_models, get_lr_sweep
)

d = get_depth_comparison()
s = get_stability_grid()
b = get_big_models()
l = get_lr_sweep()

print(f"depth_comparison: {len(d['models'])} models")
for m in d["models"]:
    div = " DIVERGED" if m.get("diverged") else ""
    vl = f"{m['final_val_loss']:.4f}" if not m.get("diverged") else "INF"
    lr = m.get("config", {}).get("learning_rate", "?")
    nl = m.get("config", {}).get("num_layers", "?")
    print(f"  L{nl} LR={lr} val={vl}{div}")

print(f"\nstability_grid: {len(s['models'])} models")
divs = [m["label"] for m in s["models"] if m.get("diverged")]
valid = [m for m in s["models"] if not m.get("diverged")]
print(f"  diverged: {divs}")
if valid:
    best = min(valid, key=lambda x: x["final_val_loss"])
    print(f"  best: {best['label']} val={best['final_val_loss']:.4f}")

print(f"\nbig_models: {len(b['models'])} models")
if b["models"]:
    best = min(b["models"], key=lambda x: x["final_val_loss"])
    worst_gap = max(b["models"], key=lambda x: x["final_val_loss"] - x["final_train_loss"])
    gap = worst_gap["final_val_loss"] - worst_gap["final_train_loss"]
    print(f"  best val: {best['label']} val={best['final_val_loss']:.4f}")
    print(f"  worst overfit: {worst_gap['label']} gap=+{gap:.4f}")

print(f"\nlr_sweep: {len(l['models'])} models")
for m in l["models"]:
    div = " DIVERGED" if m.get("diverged") else ""
    vl = f"{m['final_val_loss']:.4f}" if not m.get("diverged") else "INF"
    lr = m.get("config", {}).get("learning_rate", "?")
    print(f"  LR={lr} val={vl}{div}")

# Check loss curves and generated samples
for gn, g in [("depth", d), ("stability", s), ("big", b), ("lr", l)]:
    ms = g["models"]
    with_curves = sum(1 for m in ms if m.get("loss_curve") and
                      (len(m["loss_curve"].get("train", [])) > 0 or len(m["loss_curve"].get("val", [])) > 0))
    with_samples = sum(1 for m in ms if m.get("generated_samples") and len(m["generated_samples"]) > 0)
    print(f"\n{gn}: {with_curves}/{len(ms)} have loss curves, {with_samples}/{len(ms)} have text samples")
