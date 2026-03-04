import matplotlib.pyplot as plt
import requests
import json
import os

API_URL = "http://localhost:8000/api/v1/mlp-grid/embedding"

# We want the E2 configuration
params = {
    "embedding_dim": 2,
    "hidden_size": 1024, # You can change this if you want a different H config, but H doesn't matter much for the pure embedding matrix size
    "learning_rate": 0.01
}

def fetch_and_plot():
    print(f"Fetching embedding data from {API_URL}...")
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        vocab = data['vocab']
        embedding_matrix = data['embedding_matrix']
        
        print(f"Loaded {len(vocab)} characters with embedding dimension {len(embedding_matrix[0])}")
        
        if len(embedding_matrix[0]) != 2:
            print("Warning: Expected a 2D embedding space for this plot.")
            return

        plt.figure(figsize=(14, 10))
        plt.style.use('dark_background') # Matches the V2 UI better
        
        # Plot points
        x_coords = [row[0] for row in embedding_matrix]
        y_coords = [row[1] for row in embedding_matrix]
        
        plt.scatter(x_coords, y_coords, color='#a78bfa', alpha=0.7, s=50)
        
        # Annotate points
        for i, char in enumerate(vocab):
            # Special formatting for spaces and newlines so they're visible
            display_char = char
            if char == ' ':
                display_char = '[SPACE]'
            elif char == '\n':
                display_char = '[\\n]'

            plt.annotate(
                display_char, 
                (x_coords[i], y_coords[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=12, color='white', alpha=0.9
            )
            
        plt.title(f"Learned 2D Character Embeddings (Vocab Size: {len(vocab)})", fontsize=16, pad=20, color='white')
        plt.xlabel("Dimension 0", fontsize=12, color='gray')
        plt.ylabel("Dimension 1", fontsize=12, color='gray')
        plt.grid(True, alpha=0.2, linestyle='--')
        
        # Add a subtle crosshair at origin
        plt.axhline(0, color='white', alpha=0.2, linestyle='-')
        plt.axvline(0, color='white', alpha=0.2, linestyle='-')
        
        plt.tight_layout()
        
        save_path = "c:/Users/adril/.gemini/antigravity/brain/814df3dc-f03d-46f1-bfdb-1bb25b0ba8b4/embedding_2d_plot.png"
        plt.savefig(save_path, dpi=150, facecolor='#111827', edgecolor='none')
        print(f"Plot successfully saved to {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_and_plot()
