# utils/data.py
import torch
import config 

def load_data(path):
    """Load raw text from file"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print(f"❌ Error: File not found at {path}")
        exit()

def get_batch(data, batch_size, block_size, device):
    """Generate random batch of inputs (x) and targets (y)"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)