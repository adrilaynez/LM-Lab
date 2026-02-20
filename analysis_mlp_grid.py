import torch
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from collections import defaultdict

CHECKPOINT_DIR = Path("checkpoints/mlp_grid")

def load_analysis():
    files = list(CHECKPOINT_DIR.glob("mlp_*.pt"))
    print(f"Found {len(files)} checkpoint files.")
    
    data = []
    
    for f in files:
        try:
            res = torch.load(f, map_location="cpu")
            cfg = res['config']
            snapshots = res['snapshots']
            metadata = res['metadata']
            
            # Final snapshot usually step_50000
            final_step = max([int(k.split('_')[1]) for k in snapshots.keys()])
            final_snap = snapshots[f"step_{final_step}"]
            metrics = final_snap['metrics']
            
            # Convergence speed (from scoring logic)
            conv_step = 50000
            for entry in res['metrics_log']['train_loss']:
                if entry['value'] < 2.5:
                    conv_step = entry['step']
                    break
            
            # Embedding quality
            emb_quality = metrics.get('embedding_quality', {})
            dynamics = metrics.get('dynamics', {})
            
            # Grad Stability (STD of W1 grad norm over time)
            grad_norms = [m["value"].get("W1", 0) for m in res['metrics_log']["grad_norms"]]
            grad_stability = np.std(grad_norms) if grad_norms else 0
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
            
            row = {
                "emb_dim": cfg["emb_dim"],
                "hidden_size": cfg["hidden_size"],
                "learning_rate": cfg["learning_rate"],
                "final_val_loss": metrics["val_loss"],
                "final_train_loss": metrics["train_loss"],
                "gen_gap": metrics["generalization_gap"],
                "dead_neurons": metrics["dead_neurons"],
                "score": metadata["score"],
                "conv_step": conv_step,
                "grad_stability": grad_stability,
                "avg_grad_norm": avg_grad_norm,
                "emb_entropy": emb_quality.get("entropy", 0),
                "emb_norm_mean": emb_quality.get("norm_stats", {}).get("mean", 0),
                "drift_magnitude": dynamics.get("drift_magnitude", 0),
                "cosine_sim": dynamics.get("cosine_similarity", 0)
            }
            data.append(row)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    df = pd.DataFrame(data)
    return df

def analyze_landscape(df):
    print("\n--- Global Performance Landscape ---")
    # Pivot for Val Loss
    pivot = df.groupby(["emb_dim", "hidden_size"])["final_val_loss"].mean().unstack()
    print("Mean Val Loss by Emb Dim and Hidden Size:")
    print(pivot)
    
    # Aggregates
    print("\nAggregates by Emb Dim:")
    print(df.groupby("emb_dim")["final_val_loss"].agg(["mean", "std", "min"]))
    
    print("\nAggregates by Hidden Size:")
    print(df.groupby("hidden_size")["final_val_loss"].agg(["mean", "std", "min"]))

def analyze_dynamics(df):
    print("\n--- Learning Dynamics ---")
    print("Mean Convergence Step by Learning Rate:")
    print(df.groupby("learning_rate")["conv_step"].mean())
    
    best_conv = df.sort_values("conv_step").head(5)
    print("\nTop 5 Fastest Converging Configs:")
    print(best_conv[["emb_dim", "hidden_size", "learning_rate", "conv_step", "final_val_loss"]])

def analyze_embeddings(df):
    print("\n--- Embedding Representation Analysis ---")
    print("Mean Embedding Entropy by Dimension:")
    print(df.groupby("emb_dim")["emb_entropy"].mean())
    
    print("\nMean Drift Magnitude by Hidden Size:")
    print(df.groupby("hidden_size")["drift_magnitude"].mean())

def analyze_stability(df):
    print("\n--- Stability & Optimization Health ---")
    print("Grad Stability (higher = more volatile) by LR:")
    print(df.groupby("learning_rate")["grad_stability"].mean())
    
    # Detect anomalous configs
    exploded = df[df["avg_grad_norm"] > 10.0]
    if not exploded.empty:
        print("\nExploding Gradient Risk Configs:")
        print(exploded[["emb_dim", "hidden_size", "learning_rate", "avg_grad_norm"]])

def ranking(df):
    print("\n--- Final Configuration Ranking ---")
    
    # 1. Best Generalization (lowest val loss)
    print("\nTop 5 - Best Generalization:")
    print(df.sort_values("final_val_loss").head(5)[["emb_dim", "hidden_size", "learning_rate", "final_val_loss"]])
    
    # 2. Fastest Convergence
    print("\nTop 5 - Fastest Convergence:")
    print(df.sort_values("conv_step").head(5)[["emb_dim", "hidden_size", "learning_rate", "conv_step"]])
    
    # 3. Overall Best (Composite Score)
    print("\nTop 10 - Overall Best (Leaderboard):")
    print(df.sort_values("score", ascending=False).head(10)[["emb_dim", "hidden_size", "learning_rate", "score", "final_val_loss"]])

if __name__ == "__main__":
    df = load_analysis()
    if not df.empty:
        analyze_landscape(df)
        analyze_dynamics(df)
        analyze_embeddings(df)
        analyze_stability(df)
        ranking(df)
        
        # Save aggregate for report
        df.to_csv("mlp_grid_analysis_results.csv", index=False)
        print("\nAnalysis results saved to 'mlp_grid_analysis_results.csv'")
