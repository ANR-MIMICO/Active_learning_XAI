import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_1d_histograms():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    
    methods = ["lhs", "sur_shap", "v6_sur", "v6_dyn"]
    titles = {"lhs": "LHS", "sur_shap": "SHAP-CS", "v6_sur": "IMSE-US", "v6_dyn": "Dynamic-US"}
    colors = {"lhs": "black", "sur_shap": "green", "v6_sur": "blue", "v6_dyn": "red"}
    seeds = [42, 100, 2026, 777, 12345]
    
    var_names = ["Price_Dispose", "Scarcity", "Density", "Cluster", "Km_Cost"]
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 15))
    fig.suptitle('1D Histograms per Variable (Aggregated over 5 seeds)', fontsize=16, fontweight='bold')
    
    for i, var_name in enumerate(var_names):
        ax = axes[i]
        
        data_list = []
        label_list = []
        color_list = []
        
        for m in methods:
            df_seeds = []
            for seed in seeds:
                folder_name = f"tmp_{m}_{seed}"
                csv_path = os.path.join(results_dir, folder_name, "al_database_loop_49.csv")
                if not os.path.exists(csv_path):
                    csv_path = os.path.join(results_dir, folder_name, "al_database.csv")
                if os.path.exists(csv_path):
                    df_seeds.append(pd.read_csv(csv_path))
            
            if not df_seeds: continue
            df_combined = pd.concat(df_seeds, ignore_index=True)
            
            data_list.append(df_combined.iloc[:, i].values)
            label_list.append(titles[m])
            color_list.append(colors[m])
            
        # Custom bins for each variable
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
        else:
            bins_to_use = np.linspace(0, 1, 11)
            xticks = None
            
        # Plot stacked bar chart with manual weights so each method sums to 1
        if data_list:
            weights_list = [np.ones_like(d) / len(d) for d in data_list]
            ax.hist(data_list, bins=bins_to_use, density=False, weights=weights_list, histtype='barstacked', label=label_list, color=color_list, alpha=0.8, edgecolor='black', linewidth=0.5)
            
        ax.set_title(f'Distribution of {var_name}', fontweight='bold')
        ax.set_ylabel('Proportion (Sums to 1)')
        if var_name == "Price_Dispose":
            ax.set_xticks([2, 3, 4, 5])
        elif var_name == "Km_Cost":
            ax.set_xticks(range(1, 11))
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", "1d_histograms_variables.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"Saved 1D histograms plot to {out_img}")

if __name__ == "__main__":
    plot_1d_histograms()
