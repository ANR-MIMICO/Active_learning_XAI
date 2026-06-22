import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from smt.surrogate_models import KRG
from smt.design_space import DesignSpace, FloatVariable
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def plot_pca_shap():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    
    methods = [("tmp_lhs_42", "LHS"), 
               ("tmp_sur_shap_42", "SHAP-CS"), 
               ("tmp_v6_sur_42", "IMSE-US"),
               ("tmp_v6_dyn_42", "Dynamic-US")]
               
    design_space = DesignSpace([
        FloatVariable(1.5, 6.5),
        FloatVariable(0, 1),
        FloatVariable(0, 1),
        FloatVariable(10, 40),
        FloatVariable(0.5, 11.5)
    ])
    
    all_shap_list = []
    plot_data = []
    
    print("Extracting SHAP values for all methods...")
    for folder, title in methods:
        csv_path = os.path.join(results_dir, folder, "al_database_loop_49.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(results_dir, folder, "al_database.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {title}, data not found at {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :5].values
        y = df.iloc[:, 5].values
        
        # Train surrogate to extract SHAP
        sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space)
        sm.set_training_values(X, y)
        sm.train()
        
        explainer = shap.ExactExplainer(sm.predict_values, X)
        shap_values = explainer(X).values
        
        all_shap_list.append(shap_values)
        plot_data.append((title, X, y, shap_values))
        print(f"Processed {title}")
        
    if not plot_data:
        print("No data processed.")
        return
        
    # Fit a single global PCA on all SHAP values combined to ensure the axes are identical
    global_shap = np.vstack(all_shap_list)
    scaler = StandardScaler()
    global_shap_scaled = scaler.fit_transform(global_shap)
    
    pca = PCA(n_components=2)
    pca.fit(global_shap_scaled)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle('SHAP Space PCA Projection - Visualizing Explanatory Diversity (Seed 42)', fontsize=18, fontweight='bold')
    
    for ax, (title, X, y, shap_vals) in zip(axes, plot_data):
        # Transform SHAP values to the global PCA space
        shap_proj = pca.transform(scaler.transform(shap_vals))
        
        # Plot initial 30 points (DoE)
        sc1 = ax.scatter(shap_proj[:30, 0], shap_proj[:30, 1], c=y[:30], cmap='coolwarm', vmin=0, vmax=1, 
                         alpha=0.3, label='Initial DoE (N=30)', s=40, edgecolors='grey', linewidths=0.5)
        
        # Plot added points (Active Learning phase)
        # Using a thicker edge and larger size to distinguish them
        if len(y) > 30:
            label_al = f'AL Added Points' if title != 'LHS' else 'LHS Added Points'
            sc2 = ax.scatter(shap_proj[30:, 0], shap_proj[30:, 1], c=y[30:], cmap='coolwarm', vmin=0, vmax=1, 
                             alpha=0.9, label=label_al, s=80, edgecolors='black', linewidths=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('SHAP Principal Component 1')
        if ax == axes[0]: 
            ax.set_ylabel('SHAP Principal Component 2')
            
        # Draw a line at 0 for visual reference
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)
        
    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(sc1, cax=cbar_ax, label='Target Prediction Probability (Tipping Point > 0.5)')
    
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", "tipping_points_pca_final_shap.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"Saved SHAP PCA plot to {out_img}")

if __name__ == "__main__":
    plot_pca_shap()
