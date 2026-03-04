"""Regenerate summary.json for all pedagogical groups from .pt files."""
import torch, json, time, traceback
from pathlib import Path

BASE = Path("checkpoints/pedagogical")

def pedagogical_entry(ck):
    """Mirror _pedagogical_entry from inference.py"""
    cfg = ck.get("config", {})
    meta = ck.get("metadata", {})
    ml = ck.get("metrics_log", {})
    
    # Get train/val loss
    train_loss = meta.get("final_train_loss")
    val_loss = meta.get("final_val_loss")
    
    # Fallback to metrics_log
    if train_loss is None and ml.get("train_loss"):
        e = ml["train_loss"][-1]
        train_loss = e["value"] if isinstance(e, dict) else e
    if val_loss is None and ml.get("val_loss"):
        e = ml["val_loss"][-1]
        val_loss = e["value"] if isinstance(e, dict) else e
    
    # Fallback to config
    if train_loss is None:
        train_loss = cfg.get("final_loss", cfg.get("final_train_loss"))
    if val_loss is None:
        val_loss = cfg.get("final_val_loss")
    
    # Detect divergence
    diverged = meta.get("diverged", False)
    if train_loss is not None and (train_loss != train_loss or train_loss > 1e6):  # NaN or Inf
        diverged = True
        train_loss = float('inf')
    if val_loss is not None and (val_loss != val_loss or val_loss > 1e6):
        diverged = True
        val_loss = float('inf')
    if train_loss is None and val_loss is None:
        diverged = True
        train_loss = float('inf')
        val_loss = float('inf')
    
    # Build techniques
    techniques = meta.get("techniques", {})
    if not techniques:
        techniques = {
            "init_strategy": cfg.get("init_strategy", "random"),
            "use_batchnorm": cfg.get("use_batchnorm", cfg.get("use_batch_norm", False)),
            "use_residual": cfg.get("use_residual", False),
        }
    
    # Generated samples
    generated = meta.get("generated_samples", [])
    if not generated and ml.get("text_snapshots"):
        snaps = ml["text_snapshots"]
        if snaps:
            last = snaps[-1]
            generated = [last["text"]] if isinstance(last, dict) else [str(last)]
    
    # Loss curves
    train_curve = ml.get("train_loss", [])
    val_curve = ml.get("val_loss", [])
    
    # Total params
    total_params = meta.get("total_params", cfg.get("total_parameters", 0))
    if total_params == 0:
        # Calculate from state dict
        sd = ck.get("model_state_dict", {})
        total_params = sum(p.numel() for p in sd.values() if hasattr(p, 'numel'))
    
    label = meta.get("label", "")
    
    return {
        "label": label,
        "config": cfg,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "total_params": total_params,
        "train_time_sec": meta.get("train_time_sec"),
        "diverged": diverged,
        "expected_uniform_loss": meta.get("expected_uniform_loss"),
        "techniques": techniques,
        "generated_samples": generated,
        "loss_curve": {
            "train": train_curve,
            "val": val_curve,
        },
        "grad_norms": ml.get("grad_norms", []),
        "text_snapshots": ml.get("text_snapshots", []),
    }


for subdir in sorted(BASE.iterdir()):
    if not subdir.is_dir():
        continue
    
    pt_files = sorted(subdir.glob("*.pt"))
    if not pt_files:
        print(f"SKIP {subdir.name}: no .pt files")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {subdir.name} ({len(pt_files)} .pt files)")
    print(f"{'='*60}")
    
    results = []
    for pt_file in pt_files:
        try:
            ck = torch.load(pt_file, map_location="cpu", weights_only=False)
            
            # Ensure label exists in metadata
            meta = ck.get("metadata", {})
            if not meta.get("label"):
                meta["label"] = pt_file.stem
                ck["metadata"] = meta
            
            entry = pedagogical_entry(ck)
            results.append(entry)
            
            tr = entry["final_train_loss"]
            vl = entry["final_val_loss"]
            div = entry["diverged"]
            tr_s = f"{tr:.4f}" if tr != float('inf') else "INF"
            vl_s = f"{vl:.4f}" if vl != float('inf') else "INF"
            gap = ""
            if tr != float('inf') and vl != float('inf'):
                g = vl - tr
                gap = f"  gap={g:+.4f}"
            status = " DIVERGED" if div else ""
            print(f"  OK  {pt_file.stem:<40} train={tr_s} val={vl_s}{gap}  params={entry['total_params']}{status}")
            
        except Exception as ex:
            print(f"  ERR {pt_file.stem}: {ex}")
            traceback.print_exc()
    
    # Backup old summary
    summary_path = subdir / "summary.json"
    if summary_path.exists():
        bak = subdir / "summary.json.bak2"
        if bak.exists():
            bak.unlink()
        summary_path.rename(bak)
        print(f"  Backed up old summary to {bak.name}")
    
    # Write new summary
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  Wrote {len(results)} models to summary.json")

print("\nDone!")
