"""
MLP Grid Search Precomputation
------------------------------
Refined pipeline for character-level MLP precomputation.
Features:
- Snapshotted training (saves intermediate states)
- Advanced interpretability metrics (Karpathy Ratio, Embedding Quality, etc.)
- Qualitative time-lapse generation
- Automated configuration scoring
- Strict reproducibility

Grid Strategy:
- emb_dim: [2, 3, 6, 10, 16, 32]
- hidden_size: [32, 64, 128, 256, 512, 1024]
- learning_rate: [0.2, 0.1, 0.01]
- Total runs: 6 * 6 * 3 = 108 runs (each producing 5 snapshots)
"""

import sys
import os
import torch
import time
import json
import random
import numpy as np
import torch.nn.functional as F
from pathlib import Path

# Fix import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp import MLPModel
from utils.tokenizer import CharTokenizer
from utils.data import load_data, get_batch
from api.config import CHECKPOINT_DIR, DATA_PATH, DEVICE, SEED

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

OUTPUT_DIR = Path(CHECKPOINT_DIR) / "mlp_grid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
GRID = {
    "emb_dim": [2, 3, 6, 10, 16, 32],
    "hidden_size": [32, 64, 128, 256, 512, 1024],
    "learning_rate": [0.2, 0.1, 0.01] # Objective 1.1: 3 regimes
}

SNAPSHOT_STEPS = [1000, 5000, 10000, 20000, 50000] # Objective 1.2: Snapshots
MAX_STEPS = max(SNAPSHOT_STEPS)
BATCH_SIZE = 32
CONTEXT_SIZE = 3

def get_runs():
    """Generates a full Cartesian grid for the specified hyperparameters."""
    runs = []
    for emb_dim in GRID["emb_dim"]:
        for hidden_size in GRID["hidden_size"]:
            for lr in GRID["learning_rate"]:
                runs.append({
                    "emb_dim": emb_dim,
                    "hidden_size": hidden_size,
                    "learning_rate": lr,
                    "context_size": CONTEXT_SIZE,
                    "batch_size": BATCH_SIZE
                })
    return runs

def evaluate(model, data, block_size, batch_size=1000):
    model.eval()
    with torch.no_grad():
        ix = torch.randint(len(data) - block_size, (batch_size,))
        X = torch.stack([data[i:i+block_size] for i in ix]).to(DEVICE)
        Y = torch.stack([data[i+block_size] for i in ix]).to(DEVICE)
        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y).item()
    model.train()
    return loss

def get_qualitative_samples(model, tokenizer, num_samples=5, length=30):
    model.eval()
    samples = []
    # Use a fixed prompt for all snapshots to see evolution
    prompt = torch.zeros((num_samples, CONTEXT_SIZE), dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(prompt, length)
        for i in range(num_samples):
            samples.append(tokenizer.decode(generated[i]))
    model.train()
    return samples

def train_run(run_cfg, train_data, val_data, tokenizer):
    """
    Trains a model for MAX_STEPS and saves snapshots at specified intervals.
    """
    set_seed(SEED) # Ensure reproducibility for each run start
    
    expected_uniform_loss = -torch.log(torch.tensor(1.0/tokenizer.vocab_size)).item()
    
    model = MLPModel(
        vocab_size=tokenizer.vocab_size,
        context_size=run_cfg["context_size"],
        emb_dim=run_cfg["emb_dim"],
        hidden_size=run_cfg["hidden_size"],
        seed=SEED
    ).to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=run_cfg["learning_rate"])
    
    snapshots = {}
    metrics_log = {
        "train_loss": [],
        "val_loss": [],
        "grad_norms": [],
        "dead_neurons": []
    }
    
    block_size = run_cfg["context_size"]
    batch_size = run_cfg["batch_size"]
    
    # 0. Initial Metrics
    initial_train_loss = evaluate(model, train_data, block_size)
    initial_val_loss = evaluate(model, val_data, block_size)
    
    print(f"   Initial Loss: {initial_train_loss:.4f} (Expected: {expected_uniform_loss:.4f})")
    
    # Capture model stats
    model_stats = model.get_model_stats()
    
    # Step 0 Snapshot
    internals_init = model.get_internals()
    snapshots["step_0"] = {
        "step": 0,
        "metrics": {
            "train_loss": initial_train_loss,
            "val_loss": initial_val_loss,
            "generalization_gap": initial_val_loss - initial_train_loss,
            "dead_neurons": internals_init["dead_neurons"],
            "activation_stats": internals_init["activation_stats"],
            "weight_stats": internals_init["weight_stats"],
            "grad_norms": internals_init["grad_norms"],
            "grad_health": internals_init["grad_health"],
            "embedding_quality": model.get_embedding_quality_metrics(tokenizer),
            "samples": get_qualitative_samples(model, tokenizer)
        },
        "interpretability": {
            "embedding_matrix": internals_init["embedding_matrix"].tolist(),
            "W1": internals_init["W1"].tolist(),
            "W2": internals_init["W2"].tolist()
        }
    }

    t0 = time.time()
    last_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    for step in range(1, MAX_STEPS + 1):
        # Sample batch
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        X = torch.stack([train_data[i:i+block_size] for i in ix]).to(DEVICE)
        Y = torch.stack([train_data[i+block_size] for i in ix]).to(DEVICE)
        
        # Forward & Backward
        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Logging
        if step % 100 == 0 or step in SNAPSHOT_STEPS:
            train_loss = loss.item()
            val_loss = evaluate(model, val_data, block_size)
            metrics_log["train_loss"].append({"step": step, "value": train_loss})
            metrics_log["val_loss"].append({"step": step, "value": val_loss})
            
            # Detailed metrics
            internals = model.get_internals(X)
            metrics_log["grad_norms"].append({"step": step, "value": internals["grad_norms"]})
            metrics_log["dead_neurons"].append({"step": step, "value": internals["dead_neurons"]})

        # Snapshot saving
        if step in SNAPSHOT_STEPS:
            print(f"   Step {step}: Loss {loss.item():.4f}")
            internals = model.get_internals(X)
            emb_quality = model.get_embedding_quality_metrics(tokenizer)
            dynamics = model.get_representation_dynamics(last_state_dict)
            samples = get_qualitative_samples(model, tokenizer)
            
            snapshots[f"step_{step}"] = {
                "step": step,
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "metrics": {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "generalization_gap": val_loss - train_loss,
                    "dead_neurons": internals["dead_neurons"],
                    "activation_stats": internals["activation_stats"],
                    "weight_stats": internals["weight_stats"],
                    "grad_norms": internals["grad_norms"],
                    "grad_health": internals["grad_health"],
                    "embedding_quality": emb_quality,
                    "dynamics": dynamics,
                    "samples": samples
                },
                "interpretability": {
                    "embedding_matrix": internals["embedding_matrix"].tolist(),
                    "W1": internals["W1"].tolist(),
                    "W2": internals["W2"].tolist()
                }
            }
            # Update last state for dynamics
            last_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    t1 = time.time()
    
    # Final Scoring
    final_snap = snapshots[f"step_{MAX_STEPS}"]
    score = calculate_score(final_snap, metrics_log)
    
    return {
        "config": run_cfg,
        "snapshots": snapshots,
        "metrics_log": metrics_log,
        "metadata": {
            "expected_uniform_loss": expected_uniform_loss,
            "initial_loss": initial_train_loss,
            "train_time_sec": t1 - t0,
            "score": score,
            "vocab": tokenizer.chars,
            "model_stats": model_stats
        }
    }

def calculate_score(final_snapshot, metrics_log):
    """
    Programmatic scoring system (Objective 6):
    - 40% Final Validation Loss (lower is better)
    - 20% Convergence Speed (steps to reach 2.5 loss)
    - 15% Generalization Gap (lower is better)
    - 15% Dead Neurons Percentage (lower is better)
    - 10% Gradient Stability (standard deviation of gradient norms)
    """
    metrics = final_snapshot["metrics"]
    val_loss = metrics["val_loss"]
    gen_gap = metrics["generalization_gap"]
    dead_neurons = metrics["dead_neurons"]
    
    # Convergence: find first step where train_loss < 2.5
    convergence_step = MAX_STEPS
    for entry in metrics_log["train_loss"]:
        if entry["value"] < 2.5:
            convergence_step = entry["step"]
            break
            
    # Stability: std of W1 grad norms
    grad_norms = [m["value"].get("W1", 0) for m in metrics_log["grad_norms"]]
    stability = np.std(grad_norms) if grad_norms else 1.0
    
    # Composite score (higher is better)
    # 1. Loss component (0-40) - normalized against baseline
    loss_score = max(0, 40 * (1 - (val_loss / 4.5))) # Assuming 4.5 is a terrible initial loss
    
    # 2. Convergence component (0-20)
    conv_score = 20 * (1 - (convergence_step / MAX_STEPS))
    
    # 3. Generalization (0-15)
    gen_score = max(0, 15 * (1 - abs(gen_gap) / 0.5))
    
    # 4. Dead Neurons (0-15)
    dead_score = 15 * (1 - dead_neurons)
    
    # 5. Stability (0-10)
    stab_score = max(0, 10 * (1 - stability / 0.1))
    
    total_score = loss_score + conv_score + gen_score + dead_score + stab_score
    return float(total_score)

def main():
    print(f"🚀 Starting Refined MLP Pipeline...")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Load Data
    raw_text = load_data(DATA_PATH)
    tokenizer = CharTokenizer()
    tokenizer.train(raw_text)
    
    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Data Loaded: {len(raw_text)} chars. Vocab: {tokenizer.vocab_size}")
    
    runs = get_runs()
    registry = []
    
    for i, run_cfg in enumerate(runs):
        print(f"\n[{i+1}/{len(runs)}] Config: E{run_cfg['emb_dim']} H{run_cfg['hidden_size']} LR{run_cfg['learning_rate']}")
        
        fname = f"mlp_E{run_cfg['emb_dim']}_H{run_cfg['hidden_size']}_LR{run_cfg['learning_rate']}.pt"
        save_path = OUTPUT_DIR / fname
        
        # Skip if already exists (optional, but good for resuming)
        if save_path.exists():
            print(f"   ⏩ Skipping, already exists.")
            continue

        try:
            results = train_run(run_cfg, train_data, val_data, tokenizer)
            torch.save(results, save_path)
            
            registry.append({
                "filename": fname,
                "config": run_cfg,
                "score": results["metadata"]["score"],
                "final_val_loss": results["snapshots"][f"step_{MAX_STEPS}"]["metrics"]["val_loss"]
            })
            
            # Save registry intermittently
            with open(OUTPUT_DIR / "manifest.json", 'w') as f:
                json.dump(registry, f, indent=2)
                
            print(f"   ✅ Saved (Score: {results['metadata']['score']:.2f})")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    # Final ranking
    registry.sort(key=lambda x: x["score"], reverse=True)
    with open(OUTPUT_DIR / "manifest.json", 'w') as f:
        json.dump(registry, f, indent=2)
        
    print(f"\n🎉 Pipeline Execution Complete!")

if __name__ == "__main__":
    main()
