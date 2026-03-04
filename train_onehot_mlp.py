import torch
import torch.nn.functional as F
import os
import time
from pathlib import Path
from models.mlp import MLPModel
from utils.tokenizer import CharTokenizer
from utils.data import load_data, get_batch

# Configuration for One-Hot MLP
EMB_DIM = 0
HIDDEN_SIZE = 1024
NUM_LAYERS = 3
CONTEXT_SIZE = 4
LEARNING_RATE = 0.01
BATCH_SIZE = 1024
MAX_STEPS = 5000
EVAL_INTERVAL = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = 'data/paul_graham.txt'
SEED = 1337

def estimate_loss(model, ctx, xb, yb):
    model.eval()
    with torch.no_grad():
        with ctx:
            logits, loss = model(xb, yb)
    model.train()
    return loss.item()

def main():
    torch.manual_seed(SEED)
    out_dir = Path("checkpoints/mlp_grid")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    raw_text = load_data(DATA_PATH)
    tokenizer = CharTokenizer()
    tokenizer.train(raw_text)
    vocab_size = tokenizer.vocab_size
    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # 2. Init Model
    print(f"Initializing One-Hot MLP: emb={EMB_DIM}, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, ctx={CONTEXT_SIZE} on {DEVICE}")
    model = MLPModel(
        vocab_size=vocab_size,
        context_size=CONTEXT_SIZE,
        emb_dim=EMB_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        seed=SEED
    )
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Setup Metrics tracking
    metrics_log = {
        "train_loss": [],
        "val_loss": [],
        "grad_norms": [],
        "dead_neurons": []
    }
    snapshots = {}
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    
    ctx = torch.amp.autocast(device_type=DEVICE if DEVICE != 'mps' else 'cpu') if DEVICE != 'cpu' else torch.no_grad()
    if DEVICE == 'cpu':
        ctx = torch.enable_grad() # Just dummy
    
    start_time = time.time()
    
    for step in range(MAX_STEPS + 1):
        xb, yb = get_batch(train_data, BATCH_SIZE, CONTEXT_SIZE, DEVICE)
        
        # Forward
        logits, _ = model(xb)
        target = yb[:, -1]
        loss = F.cross_entropy(logits, target)
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Track every step for loss curve matching the 50 steps
        if step % (MAX_STEPS // 50) == 0:
            train_loss = loss.item()
            xv, yv = get_batch(val_data, BATCH_SIZE, CONTEXT_SIZE, DEVICE)
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(xv)
                val_target = yv[:, -1]
                val_loss = F.cross_entropy(val_logits, val_target).item()
            model.train()
            
            metrics_log["train_loss"].append({"step": step, "value": train_loss})
            metrics_log["val_loss"].append({"step": step, "value": val_loss})
            
            print(f"Step {step}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
            
        # Snapshot
        if step == 0 or step % 1000 == 0 or step == MAX_STEPS:
            # We only really need the last snapshot for the grid, but we'll add it anyway
            internals = model.get_internals(xb)
            
            snapshots[f"step_{step}"] = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "metrics": {
                    "train_loss": train_loss if 'train_loss' in locals() else loss.item(),
                    "val_loss": val_loss if 'val_loss' in locals() else 0.0,
                    "generalization_gap": val_loss - train_loss if 'val_loss' in locals() and 'train_loss' in locals() else 0.0,
                    "dead_neurons": internals.get("dead_neurons", 0.0),
                    "embedding_quality": model.get_embedding_quality_metrics(tokenizer)
                },
                "interpretability": {
                    "vocab": tokenizer.chars,
                    "embedding_matrix": internals.get("embedding_matrix")
                }
            }
            
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # 4. Save Checkpoint
    checkpoint = {
        "config": {
            "emb_dim": EMB_DIM,
            "hidden_size": HIDDEN_SIZE,
            "learning_rate": LEARNING_RATE,
            "context_size": CONTEXT_SIZE,
            "num_layers": NUM_LAYERS,
            "batch_size": BATCH_SIZE
        },
        "metadata": {
            "vocab": tokenizer.chars,
            "initial_loss": metrics_log["val_loss"][0]["value"] if metrics_log["val_loss"] else 0.0,
            "train_time_sec": train_time,
            "model_stats": {"total_parameters": total_params},
            "score": 100.0, # Dummy score
            "expected_uniform_loss": -torch.log(torch.tensor(1.0/vocab_size)).item()
        },
        "metrics_log": metrics_log,
        "snapshots": snapshots
    }
    
    fname = f"mlp_E{EMB_DIM}_H{HIDDEN_SIZE}_LR{LEARNING_RATE}.pt"
    save_path = out_dir / fname
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")

if __name__ == '__main__':
    main()
