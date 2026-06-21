import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cireco.scripts.cireco_paper_benchmark import prepare_simulator

def plot_pca():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    
    path_lhs = os.path.join(results_dir, "LHS_seed_42.csv")
    path_v5 = os.path.join(results_dir, "tmp_v5_42", "al_database.csv")
    if not os.path.exists(path_v5): path_v5 = os.path.join(results_dir, "tmp_v5_42", "al_database_loop_49.csv")
    
    if not os.path.exists(path_lhs) or not os.path.exists(path_v5):
        print("LHS data not found, skipping PCA plot")
        return
        
    df_lhs = pd.read_csv(path_lhs)
    X_lhs = df_lhs.iloc[:, :5].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_lhs)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    # --- GET TRUE SIMULATOR FOR BACKGROUND CONTOUR ---
    simulator = prepare_simulator()
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle('PCA Projection - Visualizing Tipping Point Hunting (Seed 42)', fontsize=18, fontweight='bold')
    
    # Pre-compute background mesh
    x_min, x_max = -3, 4
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_2d)
    grid_unscaled = scaler.inverse_transform(grid_original)
    
    Z = simulator(grid_unscaled)
    Z = Z.reshape(xx.shape)
    
    methods = [("tmp_lhs_42", "LHS"), 
               ("tmp_sur_42", "Space-US"), 
               ("tmp_sur_shap_42", "SHAP-US"), 
               ("tmp_v5_42", "Dynamic-US")]
               
    for ax, (folder, title) in zip(axes, methods):
        csv_path = os.path.join(results_dir, folder, "al_database.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(results_dir, folder, "al_database_loop_49.csv")
            if not os.path.exists(csv_path):
                continue
        
        df = pd.read_csv(csv_path)
        X_all = df.iloc[:, :5].values
        X_proj = pca.transform(scaler.transform(X_all))
        
        # Plot background contour
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.4)
        # Highlight P=0.5 boundary
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        # Plot initial 30 points in grey
        ax.scatter(X_proj[:30, 0], X_proj[:30, 1], c='grey', alpha=0.6, label='Initial DoE (30 pts)', s=40, edgecolors='white')
        
        # Plot added 50 points
        if 'v4' in folder: c = 'red'
        elif 'sur_shap' in folder: c = 'green'
        elif 'sur' in folder: c = 'blue'
        else: c = 'black'
        
        ax.scatter(X_proj[30:, 0], X_proj[30:, 1], c=c, alpha=0.9, label='Added by AL (50 pts)', s=80, edgecolors='black')
                   
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Principal Component 1')
        if ax == axes[0]: ax.set_ylabel('Principal Component 2')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", "tipping_points_pca.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"Saved PCA physical cluster plot to {out_img}")

if __name__ == "__main__":
    plot_pca()
