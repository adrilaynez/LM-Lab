# config.py
import torch

# Data Settings
DATA_PATH = 'data/paul_graham.txt'
CHECKPOINT_DIR = 'checkpoints'
MODEL_FILENAME = 'bigram_pg.pt'

# Model Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8       # Context length (irrelevant for Bigram, crucial for GPT)
LEARNING_RATE = 1e-3
MAX_STEPS = 5000
EVAL_INTERVAL = 500

# System
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 1337

# Model Architecture Options: 'bigram', 'mlp', 'rnn', 'gpt'
MODEL_TYPE = 'bigram'