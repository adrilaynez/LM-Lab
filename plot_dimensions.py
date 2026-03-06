import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_per_dimension(dim, max_plots=20):
    checkpoint_path = f"C:/Projects/LM-Lab/checkpoints/mlp_advanced/v1/mlp_advanced_{dim}d_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print(f"File not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = checkpoint["metadata"]["vocab"]
    
    # Nos quedamos con la matriz final (paso 50k o el más alto)
    metrics_log = checkpoint["metrics_log"]
    embeddings_history = metrics_log.get("embeddings_history", {})
    last_step = max(embeddings_history.keys())
    emb_matrix = np.array(embeddings_history[last_step])
    
    # Formatear caracteres
    display_vocab = []
    for char in vocab:
        if char == ' ': display_vocab.append('[SPACE]')
        elif char == '\n': display_vocab.append('[\\n]')
        else: display_vocab.append(char.upper())
        
    # Limitar el número de dimensiones a plotear para modelos gigantes
    plot_dim = min(dim, max_plots)
    
    # Crear un grid dinámico para abarcar la dimensión
    num_cols = min(5, plot_dim)
    num_rows = int(np.ceil(plot_dim / num_cols))
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 6))
    title_suffix = f" (Mostrando primeras {max_plots} dims)" if dim > max_plots else ""
    fig.suptitle(f"Disección por Dimensión - Modelo {dim}D (Step {last_step}){title_suffix}", fontsize=24, color='white', y=1.02)
    fig.patch.set_facecolor('#111827')
    
    # Si axs es un solo plot, convertirlo a array para que funcione el flatten
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
        
    axs = axs.flatten()
    
    for d in range(plot_dim):
        ax = axs[d]
        ax.set_facecolor('#111827')
        
        # Extraer los valores de esta dimensión para todo el vocabulario
        vals = emb_matrix[:, d]
        
        # Ordenar los caracteres por la magnitud de su activación (de menor a mayor)
        sorted_indices = np.argsort(vals)
        sorted_vals = vals[sorted_indices]
        sorted_chars = [display_vocab[i] for i in sorted_indices]
        
        # Colores
        colors = ['#f87171' if v < 0 else '#60a5fa' for v in sorted_vals]
        
        y_pos = np.arange(len(vocab))
        ax.barh(y_pos, sorted_vals, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_chars, fontfamily='monospace', fontsize=10, color='white')
        
        ax.set_title(f"Dim {d}", color='white', fontsize=14, pad=10)
        ax.grid(True, axis='x', alpha=0.2, linestyle='--')
        ax.tick_params(axis='x', colors='gray')
        
        # Linea vertical al 0 para diferenciar activaciones positivas/negativas
        ax.axvline(0, color='white', alpha=0.4, linestyle='-')
        
    for d in range(plot_dim, len(axs)):
        fig.delaxes(axs[d])
        
    plt.tight_layout()
    save_path = f"C:/Users/adril/.gemini/antigravity/brain/12b3f4d8-a062-4462-8e9d-550220522b63/dims_{dim}d.png"
    plt.savefig(save_path, dpi=150, facecolor='#111827', edgecolor='none', bbox_inches='tight')
    print(f"Plot saved to {save_path}")

# Plotear todos los restantes, limitando los grandes a 20 dimensiones máximo para no explotar la imagen
for d in [2, 4, 6, 24, 32, 50, 128]:
    plot_per_dimension(d, max_plots=20)
