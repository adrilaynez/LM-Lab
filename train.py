import torch
import os
import config
from models import get_model_class  # <--- IMPORT THE FACTORY, NOT THE MODEL
from utils.tokenizer import CharTokenizer
from utils.data import load_data, get_batch

# 1. Setup
torch.manual_seed(config.SEED)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

# 2. Data Pipeline (Same as before)
print(f"📚 Loading data from {config.DATA_PATH}...")
raw_text = load_data(config.DATA_PATH)

tokenizer = CharTokenizer()
tokenizer.train(raw_text) # Build vocab
vocab_size = tokenizer.vocab_size # Get size

data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 3. Model Initialization (DYNAMIC)
print(f"🧠 Initializing {config.MODEL_TYPE.upper()} Model...")

# Ask the factory for the class
ModelClass = get_model_class(config.MODEL_TYPE) 

# Instantiate it (All your models must accept vocab_size in __init__)
model = ModelClass(vocab_size) 
model = model.to(config.DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# 4. Training Loop (Universal)
# This works for Bigram AND GPT because both inherit from LMEngine
print(f"🚀 Starting training loop...")

for step in range(config.MAX_STEPS):
    # get_batch works for any sequence model
    xb, yb = get_batch(train_data, config.BATCH_SIZE, config.BLOCK_SIZE, config.DEVICE)

    # Forward pass is always the same interface
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % config.EVAL_INTERVAL == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")

# 5. Saving (Dynamic Filename)
# Save as 'bigram.pt' or 'gpt.pt' automatically
filename = f"{config.MODEL_TYPE}_checkpoint.pt"
save_path = os.path.join(config.CHECKPOINT_DIR, filename)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': { # Save config so the App knows what model this is!
        'model_type': config.MODEL_TYPE,
        'vocab_size': vocab_size,
        'chars': tokenizer.chars
    }
}

torch.save(checkpoint, save_path)
print(f"✅ Saved {config.MODEL_TYPE} model to {save_path}")