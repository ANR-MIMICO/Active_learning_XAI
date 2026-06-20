import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_efficiency(metrics_csv, output_path):
    df = pd.read_csv(metrics_csv)
    # Plot the ratio of SHAP Entropy vs Input Entropy
    # We look at the cumulative gain from the first loop
    init_ent_x = df['Entropy_Input'].iloc[0]
    init_ent_s = df['Entropy_SHAP'].iloc[0]
    
    gain_x = df['Entropy_Input'] - init_ent_x
    gain_s = df['Entropy_SHAP'] - init_ent_s
    
    # Avoid div by zero
    ratio = gain_s / (gain_x + 1e-9)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df['Loop'], ratio, color='purple', lw=2, marker='o')
    plt.axhline(y=1.0, color='r', linestyle='--', label='1:1 Efficiency Ratio')
    plt.title('Dual-Space Efficiency (SHAP Gain vs Input Gain)', fontweight='bold')
    plt.xlabel('Active Learning Loop')
    plt.ylabel('Ratio: Δ Entropy SHAP / Δ Entropy Input')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved Efficiency plot to {output_path}")

def plot_tipping_points_pca(db_csv, output_path):
    df = pd.read_csv(db_csv)
    
    # The first 30 points are LHS (Initial DoE), the rest are Active Learning
    X = df.drop(columns=['Target']).values
    y = df['Target'].values
    
    # Use PCA to project inputs to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    plt.figure(figsize=(10, 7))
    # Plot all points as a background to show the landscape
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.5, label='Background Landscape')
    
    # Overlay AL points specifically
    n_doe = 30
    if len(X) > n_doe:
        plt.scatter(X_pca[n_doe:, 0], X_pca[n_doe:, 1], c='red', marker='x', s=100, linewidths=2, label='AL Acquired Points')
        
    plt.colorbar(scatter, label='Target Value')
    plt.title('Tipping Points Discovery (PCA Projection)', fontweight='bold')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved Tipping Points plot to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str, choices=["cireco", "schelling"], required=True)
    args = parser.parse_args()

    app_dir = "cireco" if args.app == "cireco" else "Schelling_ABS"
    
    metrics_file = os.path.join(app_dir, "data", "processed", "v4_al_results", "al_metrics_history.csv")
    db_dir = os.path.join(app_dir, "data", "processed", "v4_al_results")
    
    # Find the latest database
    import glob
    db_files = glob.glob(os.path.join(db_dir, "al_database_loop_*.csv"))
    if not db_files:
        print("No database files found.")
        exit(1)
        
    latest_db = max(db_files, key=os.path.getctime)

    out_eff = os.path.join(app_dir, "figures", "analysis", "dual_space_efficiency.png")
    out_tip = os.path.join(app_dir, "figures", "analysis", "tipping_points_pca.png")
    os.makedirs(os.path.dirname(out_eff), exist_ok=True)
    
    if os.path.exists(metrics_file):
        plot_efficiency(metrics_file, out_eff)
    plot_tipping_points_pca(latest_db, out_tip)
