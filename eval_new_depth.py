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
    
    print(f"\n{title} (L={cfg['num_layers']}, LR={cfg['learning_rate']}, FINAL LOSS={checkpoint['metadata']['final_val_loss']:.3f}) Saturation Profile:")
    for i, layer_out in enumerate(acts):
        total_activations = layer_out.numel()
        saturated = (layer_out.abs() > 0.95).sum().item()
        very_saturated = (layer_out.abs() > 0.99).sum().item()
        
        sat_pct = (saturated / total_activations) * 100
        vsat_pct = (very_saturated / total_activations) * 100
        mean_abs = layer_out.abs().mean().item()
        
        print(f"  Layer {i:2d} -> Mean Abs: {mean_abs:.3f} | Sat (>0.95): {sat_pct:5.1f}% | Very Sat (>0.99): {vsat_pct:5.1f}%")

if __name__ == "__main__":
    base = r"c:\Projects\LM-Lab\checkpoints\pedagogical\depth_new_comparison"
    models = ["depth_L1", "depth_L2", "depth_L3", "depth_L4", "depth_L6", "depth_L8", "depth_L10", "depth_L12", "depth_L16", "depth_L20"]
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        for m in models:
            check_saturation(os.path.join(base, f"{m}.pt"), m)
    else:
        print("Ready. Run with 'run' argument to execute.")
