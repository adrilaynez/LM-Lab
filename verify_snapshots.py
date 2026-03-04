import matplotlib.pyplot as plt
import torch
import os

CHECKPOINT_DIR = 'c:/Projects/LM-Lab/checkpoints/mlp_timelapse'

def plot_snapshots(emb_dim):
    path = os.path.join(CHECKPOINT_DIR, f"mlp_E{emb_dim}_H64_LR0.01.pt")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    print(f"Plotting for {path}...")
    ckpt = torch.load(path, map_location='cpu')
    vocab = ckpt['vocab']['chars']
    snapshots = ckpt.get('snapshots', {})
    
    if not snapshots:
        print("No snapshots found in checkpoint.")
        return
        
    steps = sorted(list(snapshots.keys()))
    print(f"Found snapshots at steps: {steps}")
    
    # We plot the first, middle and last snapshot
    steps_to_plot = [steps[0], steps[len(steps)//2], steps[-1]]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#111827')
    fig.suptitle(f"Embedding Evolution (E={emb_dim})", color='white', fontsize=16)
    
    # Simplified PCA for dimensions > 2
    def project_2d(matrix):
        if matrix.shape[1] <= 2:
            return matrix[:, :2]
        
        # Center
        mean = matrix.mean(dim=0, keepdim=True)
        centered = matrix - mean
        
        # SVD
        U, S, V = torch.svd(centered)
        return centered @ V[:, :2]

    for i, step in enumerate(steps_to_plot):
        ax = axes[i]
        mat = snapshots[step].float()
        
        # Project to 2D
        pts = project_2d(mat)
        
        ax.set_facecolor('#111827')
        ax.scatter(pts[:, 0].numpy(), pts[:, 1].numpy(), color='#a78bfa', s=50, alpha=0.7)
        
        for j, char in enumerate(vocab):
            display_char = char
            color = 'white'
            if char in 'aeiou':
                color = '#f59e0b' # Vowel
            elif char == ' ':
                display_char = '[SPACE]'
                color = '#6b7280'
            elif char == '.':
                color = '#6b7280'
                
            ax.annotate(display_char, (pts[j, 0].item(), pts[j, 1].item()), 
                       color=color, fontsize=10, alpha=0.9, xytext=(3, 3), textcoords='offset points')
                       
        ax.set_title(f"Step {step}", color='white')
        ax.axis('off')
        
    plt.tight_layout()
    
    save_path = f"c:/Users/adril/.gemini/antigravity/brain/814df3dc-f03d-46f1-bfdb-1bb25b0ba8b4/evolution_E{emb_dim}.png"
    plt.savefig(save_path, facecolor='#111827')
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_snapshots(2)
    # plot_snapshots(10)
    # plot_snapshots(32)
