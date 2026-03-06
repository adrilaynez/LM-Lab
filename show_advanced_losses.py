import torch
import glob

files = glob.glob(r'C:\Projects\LM-Lab\checkpoints\mlp_advanced\v1\*.pt')
for f in files:
    try:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        meta = ck['metadata']
        print(f"{meta['label']}: Train Loss {meta['final_train_loss']:.4f}, Val Loss {meta['final_val_loss']:.4f}")
    except Exception as e:
        print(f, e)
