import os
import sys

# Force UTF-8 encoding for Windows standard output before importing anything that prints emojis
sys.stdout.reconfigure(encoding='utf-8')

import train_pedagogical_v2

def retrain_l2_l4():
    # Override the layers to just L2 and L4
    train_pedagogical_v2.G7_N_LAYERS = [2, 4]
    
    # We want a different seed to see if they were just unlucky
    train_pedagogical_v2.SEED = 424242 
    
    print("Loading data...")
    train_data, val_data, tokenizer = train_pedagogical_v2.load_default_data()
    print("Retraining L2 and L4...")
    # skip_existing = False to force overwrite
    train_pedagogical_v2.train_group7(train_data, val_data, tokenizer, skip_existing=False, max_workers=2)

if __name__ == "__main__":
    retrain_l2_l4()
