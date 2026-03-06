"""Deep diagnosis of val_loss discrepancy."""
import json
import torch
from pathlib import Path

base = Path("checkpoints/pedagogical")

# 1. Check backup summary for old depth models
print("=" * 70)
print("1. OLD DEPTH SUMMARY (from backup)")
print("=" * 70)
bak = base / "depth_comparison" / "summary.json.bak2"
if bak.exists():
    with open(bak) as f:
        old = json.load(f)
    for m in sorted(old, key=lambda x: x["label"]):
        cfg = m.get("config", {})
        opt = cfg.get("optimizer_type", "NOT_SAVED")
        lr = cfg.get("learning_rate", "?")
        init = cfg.get("init_strategy", "?")
        bn = cfg.get("use_batchnorm", "?")
        res = cfg.get("use_residual", "?")
        vl = m["final_val_loss"]
        tl = m["final_train_loss"]
        vl_s = f"{vl:.4f}" if isinstance(vl, (int, float)) and vl != float("inf") else "INF"
        tl_s = f"{tl:.4f}" if isinstance(tl, (int, float)) and tl != float("inf") else "INF"
        div = " DIVERGED" if m.get("diverged") else ""
        print(f"  {m['label']:<20} opt={opt:<10} lr={lr} init={init} bn={bn} res={res} "
              f"train={tl_s} val={vl_s}{div}")
else:
    print("  No backup found!")

# 2. Check current depth summary
print("\n" + "=" * 70)
print("2. CURRENT DEPTH SUMMARY")
print("=" * 70)
cur = base / "depth_comparison" / "summary.json"
if cur.exists():
    with open(cur) as f:
        current = json.load(f)
    for m in sorted(current, key=lambda x: x["label"]):
        cfg = m.get("config", {})
        opt = cfg.get("optimizer_type", "NOT_SAVED")
        lr = cfg.get("learning_rate", "?")
        init = cfg.get("init_strategy", "?")
        vl = m["final_val_loss"]
        tl = m["final_train_loss"]
        vl_s = f"{vl:.4f}" if isinstance(vl, (int, float)) and vl != float("inf") else "INF"
        tl_s = f"{tl:.4f}" if isinstance(tl, (int, float)) and tl != float("inf") else "INF"
        div = " DIVERGED" if m.get("diverged") else ""
        print(f"  {m['label']:<20} opt={opt:<10} lr={lr} init={init} "
              f"train={tl_s} val={vl_s}{div}")

# 3. Check which .pt files exist
print("\n" + "=" * 70)
print("3. EXISTING .PT FILES")
print("=" * 70)
for group in sorted(base.iterdir()):
    if not group.is_dir():
        continue
    pts = sorted(group.glob("*.pt"))
    print(f"\n  {group.name}/: {len(pts)} files")
    for pt in pts:
        print(f"    {pt.name} ({pt.stat().st_size:,} bytes)")

# 4. Load an old depth model from backup summary and compare with v1 stability_grid
print("\n" + "=" * 70)
print("4. STABILITY_GRID STATUS")
print("=" * 70)
sg_dir = base / "stability_grid"
if sg_dir.exists():
    pts = list(sg_dir.glob("*.pt"))
    print(f"  Directory exists with {len(pts)} .pt files")
    sg_sum = sg_dir / "summary.json"
    if sg_sum.exists():
        with open(sg_sum) as f:
            sg_data = json.load(f)
        print(f"  summary.json has {len(sg_data)} entries")
    sg_bak = sg_dir / "summary.json.bak2"
    if sg_bak.exists():
        with open(sg_bak) as f:
            sg_bak_data = json.load(f)
        print(f"  summary.json.bak2 has {len(sg_bak_data)} entries")
else:
    print("  stability_grid directory DOES NOT EXIST!")
    # Check if there's a backup anywhere
    for f in base.glob("**/stability*"):
        print(f"  Found: {f}")

# 5. Key comparison: v1 train_pedagogical.py settings for Group 1
print("\n" + "=" * 70)
print("5. V1 SCRIPT CONFIG FOR GROUP 1 (depth_comparison)")
print("=" * 70)
print("  From train_pedagogical.py:")
print("    PEDAGOGICAL_LR = 0.01  (NOT 0.001!)")
print("    optimizer_type = 'sgd'")
print("    use_lr_decay = False")
print("    grad_clip_val = None")
print("    G1_N_LAYERS = [2, 4, 8, 12, 16]")
print()
print("  BUT old depth_L1, L3, L5, L6 have lr=0.001 in config")
print("  AND they don't have optimizer_type saved")
print("  → These were NOT trained by the v1 Group 1 function!")

# 6. Compare scale_stability models
print("\n" + "=" * 70)
print("6. SCALE_STABILITY ANALYSIS (v2)")
print("=" * 70)
ss_dir = base / "scale_stability"
if ss_dir.exists():
    with open(ss_dir / "summary.json") as f:
        ss = json.load(f)
    # Group by hidden_size and technique
    for hidden in [256, 512]:
        print(f"\n  H={hidden}:")
        for m in sorted(ss, key=lambda x: x["config"]["num_layers"]):
            cfg = m["config"]
            if cfg["hidden_size"] != hidden:
                continue
            tech = "kaiming" if not cfg.get("use_batchnorm") else "kaiming+BN+res"
            vl = m["final_val_loss"]
            tl = m["final_train_loss"]
            nl = cfg["num_layers"]
            div = " DIVERGED" if m.get("diverged") else ""
            print(f"    L={nl:>2} {tech:<20} train={tl:.4f} val={vl:.4f} gap={vl-tl:+.4f}{div}")

# 7. Data size analysis
print("\n" + "=" * 70)
print("7. DATA_SIZE ANALYSIS (v2)")
print("=" * 70)
ds_dir = base / "data_size"
if ds_dir.exists():
    with open(ds_dir / "summary.json") as f:
        ds = json.load(f)
    for m in sorted(ds, key=lambda x: x["final_val_loss"]):
        vl = m["final_val_loss"]
        tl = m["final_train_loss"]
        print(f"  {m['label']:<20} train={tl:.4f} val={vl:.4f} gap={vl-tl:+.4f}")
