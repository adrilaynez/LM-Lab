"""Diagnose val_loss discrepancy between v1 and v2 models."""
import torch
from pathlib import Path

base = Path("checkpoints/pedagogical")

print("=" * 80)
print("INVESTIGATION: Comparing old (v1) vs new (v2) depth_comparison models")
print("=" * 80)

# 1. Load depth_comparison .pt files and inspect full metadata
depth_dir = base / "depth_comparison"
for pt in sorted(depth_dir.glob("*.pt")):
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    meta = ck["metadata"]
    cfg = ck["config"]
    ml = ck.get("metrics_log", {})
    
    # Check what's in config
    opt = cfg.get("optimizer_type", "NOT_SAVED")
    lr = cfg.get("learning_rate", "NOT_SAVED")
    init = cfg.get("init_strategy", "NOT_SAVED")
    bn = cfg.get("use_batchnorm", "NOT_SAVED")
    res = cfg.get("use_residual", "NOT_SAVED")
    bs = cfg.get("batch_size", "NOT_SAVED")
    ms = cfg.get("max_steps", "NOT_SAVED")
    
    # Loss curve info
    train_curve = ml.get("train_loss", [])
    val_curve = ml.get("val_loss", [])
    grad_curve = ml.get("grad_norms", [])
    
    first_train = train_curve[0]["value"] if train_curve else "?"
    first_val = val_curve[0]["value"] if val_curve else "?"
    last_train = train_curve[-1]["value"] if train_curve else "?"
    last_val = val_curve[-1]["value"] if val_curve else "?"
    
    # Early training: first 10 loss values
    early_train = [f"{e['value']:.3f}" for e in train_curve[:5]]
    
    # Grad norms
    first_grad = grad_curve[0]["value"] if grad_curve else "?"
    max_grad = max((e["value"] for e in grad_curve), default="?") if grad_curve else "?"
    
    print(f"\n--- {pt.name} ---")
    print(f"  optimizer: {opt}  lr: {lr}  init: {init}  bn: {bn}  res: {res}")
    print(f"  batch_size: {bs}  max_steps: {ms}")
    print(f"  final_train: {meta['final_train_loss']:.4f}  final_val: {meta['final_val_loss']:.4f}")
    print(f"  first logged train: {first_train}  first val: {first_val}")
    print(f"  last logged train: {last_train}  last val: {last_val}")
    print(f"  early train values: {early_train}")
    print(f"  first_grad: {first_grad}  max_grad: {max_grad}")
    print(f"  train_curve length: {len(train_curve)}  val_curve length: {len(val_curve)}")
    
    # Check model weights statistics
    sd = ck.get("model_state_dict", {})
    for k, v in sd.items():
        if "weight" in k and v.dim() >= 2:
            print(f"  weight {k}: shape={list(v.shape)} mean={v.mean():.4f} std={v.std():.4f} abs_max={v.abs().max():.4f}")
            break  # just first layer

# 2. Compare with stability_grid none models (should match old depth models)
print("\n" + "=" * 80)
print("CROSS-REFERENCE: stability_grid 'none' models")
print("=" * 80)

stab_dir = base / "stability_grid"
if stab_dir.exists():
    for pt in sorted(stab_dir.glob("*_none.pt")):
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        meta = ck["metadata"]
        cfg = ck["config"]
        opt = cfg.get("optimizer_type", "NOT_SAVED")
        lr = cfg.get("learning_rate", "NOT_SAVED")
        print(f"  {pt.stem:<30} opt={opt} lr={lr} train={meta['final_train_loss']:.4f} val={meta['final_val_loss']:.4f}")
else:
    print("  stability_grid directory not found!")

# 3. Compare the same arch: big_H256_L4_CTX8_E16 (v1, AdamW) vs scale_H256_L4_kaiming (v2, SGD)
print("\n" + "=" * 80)
print("SAME-ARCH COMPARISON: H=256 L=4 ctx=8 emb=16")
print("=" * 80)

big_pt = base / "big_models" / "big_H256_L4_CTX8_E16.pt"
if big_pt.exists():
    ck = torch.load(big_pt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    meta = ck["metadata"]
    print(f"  big_H256_L4 (v1): opt={cfg.get('optimizer_type','?')} lr={cfg.get('learning_rate','?')} "
          f"init={cfg.get('init_strategy','?')} bn={cfg.get('use_batchnorm','?')} res={cfg.get('use_residual','?')} "
          f"train={meta['final_train_loss']:.4f} val={meta['final_val_loss']:.4f}")

scale_pt = base / "scale_stability" / "scale_H256_L4_kaiming.pt"
if scale_pt.exists():
    ck = torch.load(scale_pt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    meta = ck["metadata"]
    print(f"  scale_H256_L4 (v2): opt={cfg.get('optimizer_type','?')} lr={cfg.get('learning_rate','?')} "
          f"init={cfg.get('init_strategy','?')} bn={cfg.get('use_batchnorm','?')} res={cfg.get('use_residual','?')} "
          f"train={meta['final_train_loss']:.4f} val={meta['final_val_loss']:.4f}")

# 4. Check if v1 depth models were originally trained by stability_grid
print("\n" + "=" * 80)
print("IDENTITY CHECK: Are old depth models copies of stability_grid?")
print("=" * 80)

old_depths = ["depth_L1.pt", "depth_L3.pt", "depth_L5.pt", "depth_L6.pt"]
for name in old_depths:
    dp = depth_dir / name
    if not dp.exists():
        continue
    dck = torch.load(dp, map_location="cpu", weights_only=False)
    d_cfg = dck["config"]
    d_meta = dck["metadata"]
    
    nl = d_cfg.get("num_layers", "?")
    # Find matching stability_grid model
    sg_name = f"L{nl}_none.pt"
    sg_path = stab_dir / sg_name if stab_dir.exists() else None
    if sg_path and sg_path.exists():
        sck = torch.load(sg_path, map_location="cpu", weights_only=False)
        s_meta = sck["metadata"]
        s_cfg = sck["config"]
        match_loss = (abs(d_meta["final_val_loss"] - s_meta["final_val_loss"]) < 0.001)
        match_params = d_meta["total_params"] == s_meta["total_params"]
        
        # Compare actual weight tensors
        d_sd = dck["model_state_dict"]
        s_sd = sck["model_state_dict"]
        weights_match = all(
            torch.equal(d_sd[k], s_sd[k]) for k in d_sd if k in s_sd
        )
        
        print(f"  {name} vs {sg_name}: loss_match={match_loss} params_match={match_params} weights_match={weights_match}")
        print(f"    depth: opt={d_cfg.get('optimizer_type','?')} lr={d_cfg.get('learning_rate','?')}")
        print(f"    stab:  opt={s_cfg.get('optimizer_type','?')} lr={s_cfg.get('learning_rate','?')}")
    else:
        print(f"  {name}: no matching stability_grid model found")
