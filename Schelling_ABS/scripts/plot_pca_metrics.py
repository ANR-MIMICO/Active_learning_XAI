import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Schelling_ABS.scripts.paper_benchmark import prepare_simulator

def plot_pca():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    
    methods = [
        ("tmp_lhs_42", "LHS"),
        ("tmp_sur_42", "Space-US"),
        ("tmp_sur_shap_42", "SHAP-US"),
        ("tmp_v5_42", "Dynamic-US")
    ]
    
    lhs_file = os.path.join(results_dir, "tmp_lhs_42", "al_database.csv")
    if not os.path.exists(lhs_file):
        print("LHS data not found, skipping PCA plot")
        return
        
    df_lhs = pd.read_csv(lhs_file)
    X_lhs = df_lhs.iloc[:, :5].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_lhs)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    # --- GET TRUE SIMULATOR FOR BACKGROUND CONTOUR ---
    mlp, mlp_scaler = prepare_simulator()
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle('PCA Projection - Visualizing Tipping Point Hunting (Seed 42)', fontsize=18, fontweight='bold')
    
    # Pre-compute background mesh
    x_min, x_max = -3, 4
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    # Inverse transform to 5D scaled, then inverse to 5D real, then feed to simulator
    grid_5d_scaled = pca.inverse_transform(grid_2d)
    grid_5d_real = scaler.inverse_transform(grid_5d_scaled)
    Z = mlp.predict_proba(mlp_scaler.transform(grid_5d_real))[:, 1]
    Z = Z.reshape(xx.shape)
    
    methods = [("lhs", "LHS"), 
               ("sur", "Space-US"), 
               ("sur_shap", "SHAP-US"), 
               ("v5", "Hybrid-US (V5)")]
               
    seeds = [42, 100, 2026, 777, 12345]
    seed_colors = ['magenta', 'cyan', 'lime', 'orange', 'purple']
               
    for ax, (m, title) in zip(axes, methods):
        # Plot background contour
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.4)
        # Highlight P=0.5 boundary
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        for seed, s_color in zip(seeds, seed_colors):
            folder = f"tmp_{m}_{seed}"
            csv_path = os.path.join(results_dir, folder, "al_database_loop_49.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(results_dir, folder, "al_database.csv")
            if not os.path.exists(csv_path):
                continue
            
            df = pd.read_csv(csv_path)
            X_all = df.iloc[:, :5].values
            X_proj = pca.transform(scaler.transform(X_all))
            
            # Plot initial 30 points in grey
            if seed == seeds[0]:
                ax.scatter(X_proj[:30, 0], X_proj[:30, 1], c='grey', alpha=0.3, label='Initial DoE', s=30, edgecolors='white')
            
            # Plot added 50 points
            ax.scatter(X_proj[30:, 0], X_proj[30:, 1], c=s_color, alpha=0.8, label=f'Seed {seed}' if m == "lhs" else "", s=50, edgecolors='black', linewidths=0.5)
                   
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
