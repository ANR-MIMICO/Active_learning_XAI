import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_hybrid_diff():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    
    var_names = ["Price_Dispose", "Scarcity", "Density", "Cluster", "Km_Cost"]
    
    seeds = [42, 100, 2026, 777, 12345]
    
    df_lhs_list = []
    df_v4_list = []
    for seed in seeds:
        path_lhs = os.path.join(results_dir, f"tmp_lhs_{seed}", "al_database_loop_49.csv")
        if not os.path.exists(path_lhs): path_lhs = os.path.join(results_dir, f"LHS_seed_{seed}.csv")
        
        path_v4 = os.path.join(results_dir, f"tmp_v5_{seed}", "al_database_loop_49.csv")
        if not os.path.exists(path_v4): path_v4 = os.path.join(results_dir, f"tmp_v5_{seed}", "al_database.csv")
        
        if os.path.exists(path_lhs) and os.path.exists(path_v4):
            df_lhs_list.append(pd.read_csv(path_lhs))
            df_v4_list.append(pd.read_csv(path_v4))
            
    if not df_lhs_list or not df_v4_list:
        print("Missing data for difference plot")
        return
        
    df_lhs = pd.concat(df_lhs_list, ignore_index=True)
    df_v4 = pd.concat(df_v4_list, ignore_index=True)
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 15))
    fig.suptitle('Targeted Search Profile ($\Delta$ Density: Hybrid-US minus LHS)', fontsize=16, fontweight='bold')
    
    for i, var_name in enumerate(var_names):
        ax = axes[i]
        
        # Custom bins for each variable
        X_lhs = df_lhs.iloc[:, i].values
        X_v5 = df_v4.iloc[:, i].values
        
        if var_name == "Price_Dispose":
            bins_to_use = np.linspace(0, 200, 11)
            xticks = np.linspace(0, 200, 5)
        elif var_name == "Scarcity":
            bins_to_use = np.linspace(-2, 2, 11)
            xticks = np.linspace(-2, 2, 5)
        elif var_name == "Density":
            bins_to_use = np.linspace(-4, -1, 11)
            xticks = np.linspace(-4, -1, 4)
        elif var_name == "Cluster":
            bins_to_use = np.linspace(0, 0.5, 11)
            xticks = np.linspace(0, 0.5, 6)
        elif var_name == "Km_Cost":
            bins_to_use = np.linspace(0, 10, 11)
            xticks = np.linspace(0, 10, 6)
            
        hist_lhs, bin_edges = np.histogram(X_lhs, bins=bins_to_use, density=True)
        hist_v4, _ = np.histogram(X_v5, bins=bins_to_use, density=True)
        
        delta = hist_v4 - hist_lhs
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        
        # Color coding: Green for over-explored (positive), Red for under-explored (negative)
        colors = ['#2ca02c' if d > 0 else '#d62728' for d in delta]
        
        ax.bar(bin_centers, delta, width=bin_widths*0.9, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
        
        ax.set_title(f'Preference Profile for {var_name}', fontweight='bold')
        ax.set_ylabel('$\Delta$ Density')
        
        if xticks is not None:
            ax.set_xticks(xticks)

        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", "hybrid_diff_profile.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"Saved Difference profile plot to {out_img}")

if __name__ == "__main__":
    plot_hybrid_diff()
