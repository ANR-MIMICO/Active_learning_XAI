import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_histograms():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    
    var_names = ["Price_Dispose", "Scarcity", "Density", "Cluster_Spread", "Km_Cost"]
    
    # 1. Load data
    try:
        df_lhs = pd.read_csv(os.path.join(results_dir, "LHS_seed_42.csv"))
        df_sur = pd.read_csv(os.path.join(results_dir, "tmp_sur_42", "al_database_loop_49.csv"))
        df_v5 = pd.read_csv(os.path.join(results_dir, "tmp_v5_42", "al_database_loop_49.csv"))
        df_sur_shap = pd.read_csv(os.path.join(results_dir, "tmp_sur_shap_42", "al_database_loop_49.csv"))
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 1D Histograms Plot ---
    fig1, axes1 = plt.subplots(1, 5, figsize=(25, 5))
    fig1.suptitle('1D Histograms of Sampled Variables (Density)', fontsize=16, fontweight='bold')
    
    for i, var_name in enumerate(var_names):
        ax = axes1[i]
        X_lhs = df_lhs.iloc[:, i].values
        X_sur = df_sur.iloc[:, i].values
        X_v5 = df_v5.iloc[:, i].values
        X_sur_shap = df_sur_shap.iloc[:, i].values
        
        min_v = min(np.min(X_lhs), np.min(X_sur), np.min(X_v5), np.min(X_sur_shap))
        max_v = max(np.max(X_lhs), np.max(X_sur), np.max(X_v5), np.max(X_sur_shap))
        bins_to_use = np.linspace(min_v, max_v, 15)
        
        ax.hist(X_lhs, bins=bins_to_use, alpha=0.3, label='LHS (Baseline)', color='black', density=True)
        ax.hist(X_sur, bins=bins_to_use, alpha=0.5, label='Space-US', color='blue', density=True, histtype='step', linewidth=2)
        ax.hist(X_v5, bins=bins_to_use, alpha=0.6, label='Dynamic-US', color='red', density=True, histtype='step', linewidth=2.5)
        ax.hist(X_sur_shap, bins=bins_to_use, alpha=0.5, label='SHAP-US', color='green', density=True, histtype='step', linewidth=2)
        
        ax.set_title(var_name, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.legend(fontsize=10)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "analysis", "1d_histograms_variables.png"))
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    fig1.savefig(out1, dpi=300)

    # --- Hybrid Diff Profile Plot ---
    fig2, axes2 = plt.subplots(5, 1, figsize=(10, 15))
    fig2.suptitle('Targeted Search Profile ($\Delta$ Density: Dynamic-US minus LHS)', fontsize=16, fontweight='bold')
    
    for i, var_name in enumerate(var_names):
        ax = axes2[i]
        X_lhs = df_lhs.iloc[:, i].values
        X_v5 = df_v5.iloc[:, i].values
        
        min_v = min(np.min(X_lhs), np.min(X_v5))
        max_v = max(np.max(X_lhs), np.max(X_v5))
        bins_to_use = np.linspace(min_v, max_v, 15)
        
        hist_lhs, bin_edges = np.histogram(X_lhs, bins=bins_to_use, density=True)
        hist_v5, _ = np.histogram(X_v5, bins=bins_to_use, density=True)
        
        delta = hist_v5 - hist_lhs
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        
        colors = ['#2ca02c' if d > 0 else '#d62728' for d in delta]
        ax.bar(bin_centers, delta, width=bin_widths*0.9, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
        ax.set_title(f'Preference Profile for {var_name}', fontweight='bold')
        ax.set_ylabel('$\Delta$ Density')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "analysis", "hybrid_diff_profile.png"))
    fig2.savefig(out2, dpi=300)
    
    print("Histograms generated successfully!")

if __name__ == "__main__":
    plot_histograms()
