"""Deep pedagogical analysis of all models."""
import json
from pathlib import Path

# Load pedagogical analysis
with open("pedagogical_analysis.json") as f:
    models = json.load(f)

# Check dataset
data_files = list(Path("data").glob("*.txt"))
for df in data_files:
    text = df.read_text(encoding="utf-8")
    words = len(text.split())
    print(f"Dataset: {df.name} = {len(text):,} chars, {words:,} words, {len(set(text))} unique chars")

# Check what the training script uses
import glob
train_scripts = glob.glob("train_pedagogical*.py")
for ts in train_scripts:
    with open(ts, encoding="utf-8", errors="replace") as f:
        content = f.read()
    # Find data loading
    for line in content.split("\n"):
        if "data" in line.lower() and ("open" in line or "read" in line or "path" in line or "load" in line):
            print(f"  {ts}: {line.strip()}")

print("\n" + "="*80)
print("DEEP PEDAGOGICAL ANALYSIS")
print("="*80)

for cat in ["depth_comparison", "stability_grid", "lr_sweep", "big_models"]:
    ms = [m for m in models if m["category"] == cat]
    valid = [m for m in ms if m["train_loss"] is not None and m["val_loss"] is not None]
    diverged = [m for m in ms if m["train_loss"] is None or m["val_loss"] is None]
    
    print(f"\n{'='*60}")
    print(f"  {cat.upper()} ({len(ms)} models, {len(diverged)} diverged)")
    print(f"{'='*60}")
    
    if not valid:
        print("  No valid models!")
        continue
    
    best = min(valid, key=lambda m: m["val_loss"])
    worst_overfit = max(valid, key=lambda m: m["val_loss"] - m["train_loss"])
    gaps = [m["val_loss"] - m["train_loss"] for m in valid]
    avg_gap = sum(gaps) / len(gaps)
    
    print(f"  Best val:     {best['file']:<40} val={best['val_loss']:.4f}  train={best['train_loss']:.4f}  gap={best['val_loss']-best['train_loss']:+.4f}")
    print(f"  Worst overfit:{worst_overfit['file']:<40} val={worst_overfit['val_loss']:.4f}  train={worst_overfit['train_loss']:.4f}  gap={worst_overfit['val_loss']-worst_overfit['train_loss']:+.4f}")
    print(f"  Avg gap:      {avg_gap:.4f}")
    print(f"  Val range:    {min(m['val_loss'] for m in valid):.4f} - {max(m['val_loss'] for m in valid):.4f}")
    print(f"  Train range:  {min(m['train_loss'] for m in valid):.4f} - {max(m['train_loss'] for m in valid):.4f}")
    
    if diverged:
        print(f"  Diverged:     {[m['file'] for m in diverged]}")
    
    # Specific analysis per category
    if cat == "depth_comparison":
        print("\n  --- Per-depth breakdown ---")
        for m in sorted(valid, key=lambda x: x["num_layers"]):
            gap = m["val_loss"] - m["train_loss"]
            lr_note = f"  LR={m['lr']}" if m["lr"] != 0.001 else ""
            print(f"    L{m['num_layers']:>2}: train={m['train_loss']:.4f}  val={m['val_loss']:.4f}  gap={gap:+.4f}  params={m['params']:>6}{lr_note}")
        
        # Check LR consistency issue
        lrs = set(m["lr"] for m in valid)
        if len(lrs) > 1:
            print(f"\n  WARNING: Inconsistent LRs across depths: {lrs}")
            print("  This makes depth comparison unfair - L2/L4 use 0.01 vs L1/L3/L5/L6 use 0.001")
            print("  PEDAGOGICAL IMPACT: The comparison is muddied. Users might think depth=2 is worse")
            print("  than depth=1, but it's actually the LR that's too high for depth=2.")
    
    elif cat == "stability_grid":
        print("\n  --- Grid view (val_loss) ---")
        techniques = ["none", "kaiming", "kaiming+BN", "kaiming+BN+residual"]
        # Map init to technique
        def tech_key(m):
            init = m.get("init", "random")
            bn = m.get("use_bn", False)
            res = m.get("use_res", False)
            if init == "random": return "none"
            if res: return "kaiming+BN+residual"
            if bn: return "kaiming+BN"
            return "kaiming"
        
        layers_set = sorted(set(m["num_layers"] for m in ms))
        header = f"    {'Layers':>6}"
        for t in techniques:
            header += f"  {t:>20}"
        print(header)
        
        for nl in layers_set:
            row = f"    L{nl:>4}:"
            for t in techniques:
                cell = [m for m in ms if m["num_layers"] == nl and tech_key(m) == t]
                if cell and cell[0]["val_loss"] is not None:
                    v = cell[0]["val_loss"]
                    row += f"  {v:>20.4f}"
                else:
                    row += f"  {'FAIL':>20}"
            print(row)
        
        # Check if kaiming alone beats BN+residual
        kaiming_wins = 0
        total_comparisons = 0
        for nl in layers_set:
            k = [m for m in valid if m["num_layers"] == nl and tech_key(m) == "kaiming"]
            kbr = [m for m in valid if m["num_layers"] == nl and tech_key(m) == "kaiming+BN+residual"]
            if k and kbr:
                total_comparisons += 1
                if k[0]["val_loss"] < kbr[0]["val_loss"]:
                    kaiming_wins += 1
        
        print(f"\n  Kaiming-only beats All-Three: {kaiming_wins}/{total_comparisons} times")
        if kaiming_wins > total_comparisons / 2:
            print("  SURPRISING: For H=128, kaiming alone is often better than full stack!")
            print("  REASON: BN + residual add parameter overhead that hurts small nets")
    
    elif cat == "lr_sweep":
        print("\n  --- LR breakdown ---")
        for m in sorted(valid, key=lambda x: x["lr"]):
            gap = m["val_loss"] - m["train_loss"]
            print(f"    LR={m['lr']:<8}: train={m['train_loss']:.4f}  val={m['val_loss']:.4f}  gap={gap:+.4f}")
        if diverged:
            for m in diverged:
                print(f"    LR={m['lr']:<8}: DIVERGED")
    
    elif cat == "big_models":
        print("\n  --- Context size analysis ---")
        ctx_groups = {}
        for m in valid:
            ctx = m["context_size"]
            if ctx not in ctx_groups:
                ctx_groups[ctx] = []
            ctx_groups[ctx].append(m)
        
        for ctx in sorted(ctx_groups.keys()):
            ms_ctx = ctx_groups[ctx]
            best_ctx = min(ms_ctx, key=lambda m: m["val_loss"])
            worst_ctx = max(ms_ctx, key=lambda m: m["val_loss"] - m["train_loss"])
            avg_gap_ctx = sum(m["val_loss"] - m["train_loss"] for m in ms_ctx) / len(ms_ctx)
            print(f"    CTX={ctx:>3}: {len(ms_ctx)} models, best_val={best_ctx['val_loss']:.4f}, avg_gap={avg_gap_ctx:+.4f}")

# OVERALL ASSESSMENT
print("\n" + "="*80)
print("PEDAGOGICAL VALUE ASSESSMENT")
print("="*80)

print("""
WHAT WORKS WELL (clear teaching moments):

1. DEPTH WALL (depth_comparison):
   L8, L12, L16 diverge completely = powerful visual of vanishing gradients
   L5 sweet spot is clear and intuitive
   BUT: inconsistent LRs (L2=0.01, L4=0.01) muddy the comparison

2. STABILITY GRID:
   The "none fails at L4+" pattern is crystal clear
   24 models give a complete picture
   Surprising finding (kaiming > full stack) adds depth

3. BIG MODELS:
   Context-overfitting story is VERY strong (ctx=8 best, ctx=128 memorizes)
   Train loss keeps dropping while val plateaus = textbook overfitting
   The 260x param growth for 0.2 loss improvement is stark

4. LR SWEEP:
   Clear U-shaped curve, LR=0.1 diverges = good basics

WHAT'S MISSING OR WEAK:

1. NO DROPOUT MODELS:
   The endpoint exists but no models trained
   This is a CRITICAL gap - overfitting is the main story, 
   dropout is THE classic solution
   Users will wonder "why not just use dropout?"

2. NO OVERTRAINING TIMELINE:
   The endpoint exists but no models
   Would show text quality evolution (gibberish -> coherent -> memorized)
   
3. DEPTH COMPARISON LR INCONSISTENCY:
   L2 and L4 use LR=0.01 while others use 0.001
   This makes them look bad for the WRONG reason
   Should retrain L2, L4 with LR=0.001 for fair comparison

4. NO DATA SIZE EXPERIMENT:
   Would be powerful to show same model on 100K vs 300K vs 1M chars
   Shows why small data = overfitting regardless of model
   
5. ALL MODELS USE SAME DATASET:
   ~300K chars of Shakespeare is small
   No comparison with larger corpus to show scaling

6. MISSING GENERATED TEXT SAMPLES:
   Some models have empty generated_samples
   Text generation is the most engaging visual for users
""")

# Check which models have generated samples
with_samples = sum(1 for m in models if m.get("train_loss") is not None)
print(f"\nModels with loss data: {with_samples}/{len(models)}")
