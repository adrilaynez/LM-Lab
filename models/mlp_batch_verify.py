"""
Verification script for the MLP precomputation pipeline.
Tests:
1. Metric collection completeness.
2. Snapshot integrity.
3. Strict reproducibility.
"""

import sys
import os
import torch
import shutil
from pathlib import Path

# Fix import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp_precompute import train_run, set_seed
from utils.tokenizer import CharTokenizer
from utils.data import load_data
from api.config import DATA_PATH, SEED

def verify():
    print("🔍 Starting Verification...")
    
    # 1. Setup
    raw_text = load_data("data/paul_graham.txt")
    tokenizer = CharTokenizer()
    tokenizer.train(raw_text)
    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    run_cfg = {
        "emb_dim": 2,
        "hidden_size": 32,
        "learning_rate": 0.1,
        "context_size": 3,
        "batch_size": 16,
        "steps": 1000 # Short run for verification
    }
    
    # Overriding SNAPSHOT_STEPS for faster test
    import models.mlp_precompute
    original_snapshots = models.mlp_precompute.SNAPSHOT_STEPS
    original_max = models.mlp_precompute.MAX_STEPS
    models.mlp_precompute.SNAPSHOT_STEPS = [100, 500, 1000]
    models.mlp_precompute.MAX_STEPS = 1000
    
    print("--- Test 1: Metric Collection & Snapshots ---")
    results1 = train_run(run_cfg, train_data, val_data, tokenizer)
    
    # Check for Step 0
    if "step_0" not in results1["snapshots"]:
        print("❌ Missing step_0 snapshot")
        return
    print("✅ Step 0 snapshot found.")

    # Check for Model Stats
    if "model_stats" not in results1["metadata"]:
        print("❌ Missing model_stats in metadata")
        return
    mstats = results1["metadata"]["model_stats"]
    if "total_parameters" not in mstats or "inference_latency_ms" not in mstats:
        print(f"❌ Incomplete model_stats: {mstats}")
        return
    print(f"✅ Model stats verified: {mstats['total_parameters']} params, {mstats['inference_latency_ms']:.4f}ms latency")

    # Check snapshots
    expected_snaps = ["step_0", "step_100", "step_500", "step_1000"]
    for s in expected_snaps:
        if s not in results1["snapshots"]:
            print(f"❌ Missing snapshot: {s}")
            return
        snap = results1["snapshots"][s]
        required_metrics = ["train_loss", "val_loss", "dead_neurons", "activation_stats", "weight_stats", "grad_norms", "grad_health", "samples"]
        if s != "step_0":
             required_metrics.append("embedding_quality")
             required_metrics.append("dynamics")
             
        for m in required_metrics:
            if m not in snap["metrics"]:
                print(f"❌ Missing metric '{m}' in snapshot {s}")
                return
    print("✅ Snapshots and metrics verified.")

    print("\n--- Test 2: Reproducibility ---")
    results2 = train_run(run_cfg, train_data, val_data, tokenizer)
    
    loss1 = results1["snapshots"]["step_1000"]["metrics"]["train_loss"]
    loss2 = results2["snapshots"]["step_1000"]["metrics"]["train_loss"]
    
    print(f"Run 1 Final Loss: {loss1:.8f}")
    print(f"Run 2 Final Loss: {loss2:.8f}")
    
    if abs(loss1 - loss2) < 1e-7:
        print("✅ Strict reproducibility confirmed.")
    else:
        print("❌ Reproducibility failure!")
        return

    print("\n--- Test 3: Qualitative Samples ---")
    samples = results1["snapshots"]["step_1000"]["metrics"]["samples"]
    print(f"Sample generated text: {samples[0]}")
    if len(samples) == 5 and all(isinstance(s, str) for s in samples):
        print("✅ Qualitative samples verified.")
    else:
        print("❌ Qualitative samples failed!")
        return

    # Restore originals
    models.mlp_precompute.SNAPSHOT_STEPS = original_snapshots
    models.mlp_precompute.MAX_STEPS = original_max
    
    print("\n🎉 ALL VERIFICATION TESTS PASSED!")

if __name__ == "__main__":
    verify()
