"""Analyze all pedagogical models and output JSON."""
import torch, json
from pathlib import Path

BASE = Path("checkpoints/pedagogical")
results = []

for subdir in sorted(BASE.iterdir()):
    if not subdir.is_dir():
        continue
    category = subdir.name
    for f in sorted(subdir.glob("*.pt")):
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            cfg = ck.get("config", {})
            snaps = ck.get("snapshots", {})
            sorted_steps = sorted(snaps.keys(), key=lambda s: int(s.split("_")[1])) if snaps else []
            
            # Get metrics from last snapshot
            last_metrics = {}
            if sorted_steps:
                last_metrics = snaps[sorted_steps[-1]].get("metrics", {})
            
            # Get metrics log
            ml = ck.get("metrics_log", {})
            
            # Train/val from last snapshot metrics
            tr = last_metrics.get("train_loss")
            vl = last_metrics.get("val_loss")
            
            # Fallback to config
            if tr is None:
                tr = cfg.get("final_loss") or cfg.get("final_train_loss")
            if vl is None:
                vl = cfg.get("final_val_loss")
            
            # Fallback to metrics_log last entry
            if tr is None and ml.get("train_loss"):
                e = ml["train_loss"][-1]
                tr = e["value"] if isinstance(e, dict) else e
            if vl is None and ml.get("val_loss"):
                e = ml["val_loss"][-1]
                vl = e["value"] if isinstance(e, dict) else e
            
            # Get grad norms and dead neurons from metrics_log
            grad_norms = ml.get("grad_norms", [])
            dead_neurons = ml.get("dead_neurons", [])
            
            last_grad = None
            if grad_norms:
                e = grad_norms[-1]
                last_grad = e["value"] if isinstance(e, dict) else e
            
            last_dead = None
            if dead_neurons:
                e = dead_neurons[-1]
                last_dead = e["value"] if isinstance(e, dict) else e
            
            results.append({
                "file": f.stem,
                "category": category,
                "emb_dim": cfg.get("embedding_dim", cfg.get("emb_dim", 0)),
                "hidden_size": cfg.get("hidden_size", 0),
                "num_layers": cfg.get("num_layers", 1),
                "context_size": cfg.get("context_size", 4),
                "lr": cfg.get("learning_rate", 0),
                "dropout": cfg.get("dropout", 0.0),
                "params": cfg.get("total_parameters", 0),
                "init": cfg.get("init_strategy", ""),
                "use_bn": cfg.get("use_batch_norm", False),
                "use_res": cfg.get("use_residual", False),
                "train_loss": round(tr, 4) if tr else None,
                "val_loss": round(vl, 4) if vl else None,
                "last_grad_norm": round(last_grad, 6) if last_grad else None,
                "last_dead_neurons": last_dead,
                "n_snapshots": len(sorted_steps),
                "train_steps": int(sorted_steps[-1].split("_")[1]) if sorted_steps else 0,
                "all_config_keys": list(cfg.keys()),
            })
        except Exception as ex:
            print(f"ERR {f.name}: {ex}")

with open("pedagogical_analysis.json", "w") as fp:
    json.dump(results, fp, indent=2)

# Print summary per category
for cat in sorted(set(r["category"] for r in results)):
    models = [r for r in results if r["category"] == cat]
    print(f"\n{'='*80}")
    print(f"CATEGORY: {cat} ({len(models)} models)")
    print(f"{'='*80}")
    for m in models:
        gap = ""
        if m["train_loss"] and m["val_loss"]:
            g = m["val_loss"] - m["train_loss"]
            gap = f"  gap={g:+.4f}"
        tr_s = f"{m['train_loss']:.4f}" if m["train_loss"] else "N/A"
        vl_s = f"{m['val_loss']:.4f}" if m["val_loss"] else "N/A"
        extras = []
        if m["init"]:
            extras.append(f"init={m['init']}")
        if m["use_bn"]:
            extras.append("BN")
        if m["use_res"]:
            extras.append("residual")
        if m["dropout"] > 0:
            extras.append(f"drop={m['dropout']}")
        if m["last_grad_norm"] is not None:
            extras.append(f"grad={m['last_grad_norm']:.4f}")
        if m["last_dead_neurons"] is not None:
            extras.append(f"dead={m['last_dead_neurons']}")
        ext = "  [" + ", ".join(extras) + "]" if extras else ""
        print(f"  {m['file']:<45} L={m['num_layers']} H={m['hidden_size']} E={m['emb_dim']} CTX={m['context_size']} LR={m['lr']}  train={tr_s} val={vl_s}{gap}{ext}")

print(f"\nTotal: {len(results)} pedagogical models")
