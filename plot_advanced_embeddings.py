import torch
import matplotlib.pyplot as plt
import os

checkpoint_path_base = r"C:\Projects\LM-Lab\checkpoints\mlp_advanced\v1\mlp_advanced_{}d_checkpoint.pt"

def plot_evolution(dim):
    checkpoint_path = checkpoint_path_base.format(dim)
    if not os.path.exists(checkpoint_path):
        print(f"File not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    vocab = checkpoint["metadata"]["vocab"]
    metrics_log = checkpoint["metrics_log"]
    embeddings_history = metrics_log.get("embeddings_history", {})

    steps = sorted(embeddings_history.keys())
    print(f"Found steps: {steps} for {dim}D")

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Evolución de los Embeddings {dim}D", fontsize=20, color='white')
    fig.patch.set_facecolor('#111827')

    axs = axs.flatten()

    for i, step in enumerate(steps[:6]):
        ax = axs[i]
        ax.set_facecolor('#111827')
        emb_matrix = embeddings_history[step]
        
        # We only plot the first 2 dimensions even if it's 4D, 6D, etc. to see what happens
        x_coords = [row[0] for row in emb_matrix]
        y_coords = [row[1] for row in emb_matrix]
        
        ax.scatter(x_coords, y_coords, color='#a78bfa', alpha=0.7, s=50)
        
        for j, char in enumerate(vocab):
            display_char = char
            if char == ' ': display_char = '[SPACE]'
            elif char == '\n': display_char = '[\\n]'

            ax.annotate(
                display_char, 
                (x_coords[j], y_coords[j]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, color='white', alpha=0.9
            )
            
        ax.set_title(f"Step {step}", color='white')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.axhline(0, color='white', alpha=0.2, linestyle='-')
        ax.axvline(0, color='white', alpha=0.2, linestyle='-')

    plt.tight_layout()
    save_path = f"C:/Users/adril/.gemini/antigravity/brain/12b3f4d8-a062-4462-8e9d-550220522b63/evolution_{dim}d.png"
    plt.savefig(save_path, dpi=150, facecolor='#111827', edgecolor='none')
    print(f"Plot saved to {save_path}")

plot_evolution(2)
plot_evolution(4)
