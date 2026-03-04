import torch
import torch.nn.functional as F
import os
import time
from utils.data import load_data, get_batch
from utils.tokenizer import CharTokenizer
from models.mlp import MLPModel
from config import DEVICE, CHECKPOINT_DIR

# --- Configuration ---
DATA_PATH = 'data/tinyshakespeare_clean.txt'
CONTEXT_SIZE = 4
HIDDEN_SIZE = 64
LEARNING_RATE = 0.01
MAX_STEPS = 50000
EVAL_ITERS = 200
BATCH_SIZE = 256
EMB_DIMS = [2, 10, 32]
SNAPSHOT_STEPS = {0, 1000, 5000, 10000, 20000, 50000}

def train_model(emb_dim, train_data, val_data, vocab_size, tokenizer):
    print(f"\n{'='*50}")
    print(f"Training Model with emb_dim={emb_dim}")
    print(f"{'='*50}")
    
    model = MLPModel(vocab_size, CONTEXT_SIZE, emb_dim, HIDDEN_SIZE, num_layers=1)
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loss_history = []
    val_loss_history = []
    snapshots = {}
    
    # Force an evaluation at step 0
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split, data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(data, BATCH_SIZE, CONTEXT_SIZE, DEVICE)
                # target is the last char
                yb = Y[:, -1]
                logits, _ = model(X)
                loss = F.cross_entropy(logits, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out
        
    start_time = time.time()
    
    for step in range(MAX_STEPS + 1):
        # 1. Take snapshot if necessary
        if step in SNAPSHOT_STEPS:
            model.eval()
            print(f"  [Step {step}] Taking embedding snapshot...")
            emb_matrix = model.C.weight.data.clone().cpu()
            snapshots[step] = emb_matrix
            model.train()
            
        # 2. Evaluate
        if step % 1000 == 0 or step == MAX_STEPS:
            losses = estimate_loss()
            train_loss_history.append({"step": step, "loss": losses['train']})
            val_loss_history.append({"step": step, "loss": losses['val']})
            print(f"Step {step:5d}: Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f}")
            
        if step == MAX_STEPS:
            break
            
        # 3. Train
        Xb, Yb = get_batch(train_data, BATCH_SIZE, CONTEXT_SIZE, DEVICE)
        Yb = Yb[:, -1]
        
        logits, _ = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")
    
    final_val_loss = val_loss_history[-1]['loss']
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'context_size': CONTEXT_SIZE,
            'embedding_dim': emb_dim,
            'hidden_size': HIDDEN_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_layers': 1,
            'final_loss': final_val_loss,
            'total_parameters': total_params,
            'vocab_size': vocab_size
        },
        'vocab': {
            'stoi': tokenizer.stoi,
            'itos': tokenizer.itos,
            'chars': tokenizer.chars
        },
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'snapshots': snapshots
    }
    # Standard format expected by frontend, but saved in a separate directory
    filename = f"mlp_E{emb_dim}_H{HIDDEN_SIZE}_LR{LEARNING_RATE}.pt"
    os.makedirs(os.path.join(CHECKPOINT_DIR, "mlp_timelapse"), exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, "mlp_timelapse", filename)
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")

def main():
    print(f"Loading data from {DATA_PATH}...")
    raw_text = load_data(DATA_PATH)
    
    tokenizer = CharTokenizer()
    tokenizer.train(raw_text)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size} ({repr(''.join(tokenizer.chars))})")
    
    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    for emb_dim in EMB_DIMS:
        train_model(emb_dim, train_data, val_data, vocab_size, tokenizer)

if __name__ == "__main__":
    main()
