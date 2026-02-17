import torch
import os
import config
from models import get_model_class
from utils.tokenizer import CharTokenizer
from utils.data import load_data, get_batch

# 1. Setup
torch.manual_seed(config.SEED)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

# 2. Data Pipeline
print(f"📚 Loading data from {config.DATA_PATH}...")
raw_text = load_data(config.DATA_PATH)

tokenizer = CharTokenizer()
tokenizer.train(raw_text)
vocab_size = tokenizer.vocab_size

data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Calculate model parameters
print(f"🧠 Initializing {config.MODEL_TYPE.upper()} Model...")

ModelClass = get_model_class(config.MODEL_TYPE) 
model = ModelClass(vocab_size) 
model = model.to(config.DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"📊 Model Statistics:")
print(f"   - Total parameters: {total_params:,}")
print(f"   - Trainable parameters: {trainable_params:,}")
print(f"   - Vocabulary size: {vocab_size}")
print(f"   - Training data size: {len(train_data):,} tokens ({len(raw_text):,} characters)")
print(f"   - Validation data size: {len(val_data):,} tokens")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# 4. Training Loop
print(f"🚀 Starting training loop...")
loss_history = []

for step in range(config.MAX_STEPS):
    xb, yb = get_batch(train_data, config.BATCH_SIZE, config.BLOCK_SIZE, config.DEVICE)

    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())

    if step % config.EVAL_INTERVAL == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")

# 5. Save checkpoint with metadata
filename = f"{config.MODEL_TYPE}_checkpoint.pt"
save_path = os.path.join(config.CHECKPOINT_DIR, filename)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'model_type': config.MODEL_TYPE,
        'vocab_size': vocab_size,
        'chars': tokenizer.chars,
        'block_size': config.BLOCK_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'batch_size': config.BATCH_SIZE,
    },
    'training_info': {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'training_steps': config.MAX_STEPS,
        'final_loss': loss.item(),
        'loss_history': loss_history,
        'raw_text_size': len(raw_text),
        'train_data_size': len(train_data),
        'val_data_size': len(val_data),
        'unique_characters': len(tokenizer.chars),
    }
}

torch.save(checkpoint, save_path)
print(f"✅ Saved {config.MODEL_TYPE} model to {save_path}")
print(f"   - Final loss: {loss.item():.4f}")