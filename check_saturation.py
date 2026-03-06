import torch
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"c:\Projects\LM-Lab")

from models.mlp_deep import MLPDeepModel
import train_pedagogical_v2

def check_saturation(model_path, title):
    if not os.path.exists(model_path):
        print(f"Skipping {title}, file not found")
        return
        
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    
    model = MLPDeepModel(
        vocab_size=cfg["vocab_size"],
        context_size=cfg["context_size"],
        emb_dim=cfg["emb_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        init_strategy=cfg["init_strategy"],
        use_batchnorm=cfg["use_batchnorm"],
        use_residual=cfg["use_residual"],
        activation_func="tanh"
    ).eval()
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    train_data, val_data, tok = train_pedagogical_v2.load_default_data()
    ix = torch.randint(len(train_data) - cfg["context_size"] - 1, (100,))
    X = torch.stack([train_data[i:i + cfg["context_size"]] for i in ix])
    
    with torch.no_grad():
        _, _, acts = model(X, return_activations=True)
    
    # Calculate more granular saturation info
    print(f"\n{title} (L={cfg['num_layers']}) Saturation Profile:")
    for layer_out in acts:
        # layer_out shape: (Batch, Tokens, Hidden)
        total_activations = layer_out.numel()
        saturated = (layer_out.abs() > 0.95).sum().item()
        very_saturated = (layer_out.abs() > 0.99).sum().item()
        
        sat_pct = (saturated / total_activations) * 100
        vsat_pct = (very_saturated / total_activations) * 100
        mean_abs = layer_out.abs().mean().item()
        
        print(f"  -> Mean Abs: {mean_abs:.3f} | Sat (>0.95): {sat_pct:5.1f}% | Very Sat (>0.99): {vsat_pct:5.1f}%")

if __name__ == "__main__":
    base = r"c:\Projects\LM-Lab\checkpoints\pedagogical\depth_comparison"
    models = ["depth_L2", "depth_L4", "depth_L10", "depth_L16"]
    
    for m in models:
        check_saturation(os.path.join(base, f"{m}.pt"), m)
    
    print("\n--- Training L1 ... ---")
    train_pedagogical_v2.G7_N_LAYERS = [1]
    train_pedagogical_v2.SEED = 111111
    
    train_data, val_data, tokenizer = train_pedagogical_v2.load_default_data()
    train_pedagogical_v2.train_group7(train_data, val_data, tokenizer, skip_existing=False, max_workers=1)
    
    check_saturation(os.path.join(base, "depth_L1.pt"), "depth_L1")
