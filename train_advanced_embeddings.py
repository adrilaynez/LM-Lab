import sys
import os
import torch
import torch.nn.functional as F
import json
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Fix import path to include api and models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from models.mlp_advanced import MLPAdvancedModel
from utils.tokenizer import CharTokenizer
from utils.data import load_data
from api.config import CHECKPOINT_DIR, DATA_DIR, DEVICE, SEED

DATA_PATH = DATA_DIR / "tinyshakespeare_clean.txt"

_print_lock = threading.Lock()

def tprint(msg: str):
    """Thread-safe print.""" # encode safe
    msg = msg.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
    with _print_lock:
        print(msg, flush=True)

def get_batch(data, batch_size, context_size):
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+context_size] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def evaluate(model, data, context_size, batch_size=1024):
    model.eval()
    with torch.no_grad():
        max_start = len(data) - context_size - 1
        if max_start <= 0: return float("nan")
        ix = torch.randint(max_start, (batch_size,))
        X = torch.stack([data[i:i + context_size] for i in ix]).to(DEVICE)
        Y = torch.stack([data[i + context_size] for i in ix]).to(DEVICE)
        logits, _ = model(X)
        loss = F.cross_entropy(logits, Y).item()
    model.train()
    return loss

def get_samples(model, tokenizer, context_size, num=3, length=100):
    model.eval()
    prompt = torch.zeros((num, context_size), dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(prompt, length)
        samples = [tokenizer.decode(generated[i].tolist()) for i in range(num)]
    model.train()
    return samples

def train_advanced_model(dim, train_data, val_data, vocab_size, tokenizer):
    tprint(f"\n[START] Iniciando entrenamiento para dimension {dim}D en {DEVICE}...")
    
    # Nuevos parametros de modelo mas potentes
    context_size = 8
    batch_size = 256
    max_steps = 50000
    learning_rate = 0.003
    l1_lambda = 1e-4 
    tie_weights = False # Rompemos tie weights para permitir una red mucho mas profunda con embeddings pequenos
    
    # Snapshots points requested by user + step 0
    SNAPSHOT_STEPS = {0, 1000, 5000, 10000, 20000, 50000}

    model = MLPAdvancedModel(
        vocab_size=vocab_size, 
        context_size=context_size, 
        emb_dim=dim, 
        hidden_size=256, 
        num_layers=3,     # Red mas profunda y expresiva
        tie_weights=tie_weights,
        orthogonal_init=True
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Pedadogical variables to track
    metrics_log = {
        "train_loss": [], 
        "val_loss": [], 
        "grad_norms": [], 
        "text_snapshots": [],
        "embeddings_history": {}
    }

    model.train()
    t0 = time.time()
    
    # Take Step 0 Snapshot
    model.eval()
    with torch.no_grad():
        v_loss_0 = evaluate(model, val_data, context_size)
        snap_0 = get_samples(model, tokenizer, context_size, num=1, length=100)
        metrics_log["text_snapshots"].append({"step": 0, "text": snap_0[0]})
        metrics_log["embeddings_history"][0] = model.C.weight.cpu().numpy().tolist()
        metrics_log["val_loss"].append({"step": 0, "value": round(v_loss_0, 6)})
    model.train()
    
    for step in range(1, max_steps + 1):
        # Cosine Annealing LR inline
        min_lr = learning_rate * 0.1
        decay_ratio = step / max_steps
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        current_lr = min_lr + coeff * (learning_rate - min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        Xb, Yb = get_batch(train_data, batch_size, context_size)
        
        logits, loss = model(Xb, targets=Yb)
        
        # L1 Regularization
        l1_norm = torch.norm(model.C.weight, p=1)
        total_loss = loss + l1_lambda * l1_norm

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        
        # Calculate Grad Norm
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        
        optimizer.step()

        # Pedagogical logging interval
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                t_loss = loss.item()
                v_loss = evaluate(model, val_data, context_size)
            metrics_log["train_loss"].append({"step": step, "value": round(t_loss, 6)})
            metrics_log["val_loss"].append({"step": step, "value": round(v_loss, 6)})
            metrics_log["grad_norms"].append({"step": step, "value": round(total_grad_norm, 6)})
            model.train()

        if step % 10000 == 0 or step == max_steps:
             tprint(f"[{dim}D] Step {step:5d} | Loss: {loss.item():.4f} | L1: {l1_norm.item():.4f} | LR: {current_lr:.6f}")
            
        # Snapshot Logic
        if step in SNAPSHOT_STEPS:
            model.eval()
            with torch.no_grad():
                snap = get_samples(model, tokenizer, context_size, num=1, length=100)
            metrics_log["text_snapshots"].append({"step": step, "text": snap[0]})
            metrics_log["embeddings_history"][step] = model.C.weight.cpu().numpy().tolist()
            model.train()

    train_time = time.time() - t0
    
    # Final eval
    model.eval()
    with torch.no_grad():
        final_train = evaluate(model, train_data, context_size)
        final_val   = evaluate(model, val_data, context_size)
        samples     = get_samples(model, tokenizer, context_size)
    
    tprint(f"[OK] Entrenamiento {dim}D completado en {train_time:.1f}s. Loss final: {final_train:.4f}")

    # Guardar Checkpoint estilo LM-Lab Pedagogical (Todos en la misma subcarpeta)
    model_id = f"mlp_advanced_{dim}d"
    v1_dir = CHECKPOINT_DIR / "mlp_advanced" / "v1"
    v1_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = v1_dir / f"{model_id}_checkpoint.pt"
    
    config = {
        "model_type": "mlp_advanced",
        "vocab_size": vocab_size,
        "chars": tokenizer.chars,
        "context_size": context_size,
        "emb_dim": dim,
        "hidden_size": 256,
        "num_layers": 3,
        "tie_weights": tie_weights,
        "init_strategy": "orthogonal",
    }
    
    training_info = {
        "training_steps": max_steps,
        "final_loss": final_train,
        "dataset": "tinyshakespeare_clean.txt",
        "l1_lambda": l1_lambda,
        "lr_scheduler": "CosineAnnealingLR"
    }
    
    metadata = {
        "label": f"Advanced MLP {dim}D",
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "total_params": sum(p.numel() for p in model.parameters()),
        "train_time_sec": round(train_time, 2),
        "generated_samples": samples,
        "vocab": tokenizer.chars,
    }
    
    torch.save({
        "config": config,
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "training_info": training_info,
        "metrics_log": metrics_log,
        "metadata": metadata
    }, checkpoint_path)
    
    tprint(f"  [SAVE] Guardado en: {checkpoint_path}")
    return model

def main():
    tprint("Preparando datos y tokenizador...")
    raw_text = load_data(DATA_PATH)
    tokenizer = CharTokenizer()
    tokenizer.train(raw_text)
    vocab_size = tokenizer.vocab_size

    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    tprint(f"Vocab Size: {vocab_size} | Train Size: {len(train_data)} | Val Size: {len(val_data)}")

    # Las dimensiones que el usuario pidió entrenar para analizar
    dimensions = [2, 4, 6, 10, 16, 24, 32, 50, 128]
    
    MAX_WORKERS = 4
    tprint(f"\nIniciando entrenamiento concurrente con {MAX_WORKERS} workers para {len(dimensions)} modelos...\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_dim = {
            executor.submit(train_advanced_model, dim, train_data, val_data, vocab_size, tokenizer): dim
            for dim in dimensions
        }
        for future in as_completed(future_to_dim):
            dim = future_to_dim[future]
            try:
                future.result()
            except Exception as e:
                tprint(f"[FAILED] Error entrenando modelo {dim}D: {e}")
                import traceback
                traceback.print_exc()
        
    tprint("\nTodos los modelos concurrentes han sido entrenados y guardados correctamente.")

if __name__ == "__main__":
    main()
