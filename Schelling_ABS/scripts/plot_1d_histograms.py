import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_1d_histograms():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    
    methods = ["tmp_lhs_42", "tmp_sur_42", "tmp_sur_shap_42", "tmp_v4_42"]
    titles = {"tmp_lhs_42": "LHS", "tmp_sur_42": "Space-US", "tmp_sur_shap_42": "SHAP-US", "tmp_v4_42": "Hybrid-US"}
    colors = {"tmp_lhs_42": "black", "tmp_sur_42": "blue", "tmp_sur_shap_42": "green", "tmp_v4_42": "red"}
    
    var_names = ["Radius", "Ratio", "Tolerance", "Grid Size", "N Groups"]
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 15))
    fig.suptitle('1D Histograms per Variable (Active Learning Additions)', fontsize=16, fontweight='bold')
    
    for i, var_name in enumerate(var_names):
        ax = axes[i]
        
        data_list = []
        label_list = []
        color_list = []
        
        for m in methods:
            csv_path = os.path.join(results_dir, m, "al_database.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(results_dir, m, "al_database_loop_49.csv")
                if not os.path.exists(csv_path): continue
                
            df = pd.read_csv(csv_path)
            # Use all 80 points
            data_list.append(df.iloc[:, i].values)
            label_list.append(titles[m])
            color_list.append(colors[m])
            
        # Custom bins for each variable
        if var_name == "Radius":
            bins_to_use = np.arange(1.5, 6.5, 1) # Centers at 2, 3, 4, 5
        elif var_name == "N Groups":
            bins_to_use = np.arange(0.5, 11.5, 1) # Centers at 1, 2, ..., 10
        elif var_name == "Grid Size":
            bins_to_use = np.linspace(10, 40, 11)
        else:
            bins_to_use = np.linspace(0, 1, 11) # For Ratio and Tolerance
            
        # Plot stacked bar chart
        if data_list:
            ax.hist(data_list, bins=bins_to_use, density=True, histtype='barstacked', label=label_list, color=color_list, alpha=0.8, edgecolor='black', linewidth=0.5)
            
        ax.set_title(f'Distribution of {var_name}', fontweight='bold')
        ax.set_ylabel('Density')
        if var_name == "Radius":
            ax.set_xticks([2, 3, 4, 5])
        elif var_name == "N Groups":
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
