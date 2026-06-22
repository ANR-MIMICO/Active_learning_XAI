import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_dual_space():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    
    methods = ["LHS", "SUR_SHAP", "V6_SUR", "V6_DYN"]
    colors = {"LHS": "black", "SUR_SHAP": "green", "V6_SUR": "blue", "V6_DYN": "red"}
    labels = {
        "LHS": "LHS", 
        "SUR_SHAP": "SHAP-US",
        "V6_SUR": "IMSE-US",
        "V6_DYN": "Dynamic-US"
    }

    plt.figure(figsize=(10, 8))
    plt.title('Dual Space Efficiency (SHAP Entropy Gain vs Physical Entropy Gain)', fontsize=16, fontweight='bold')
    
    for m in methods:
        csv_files = glob.glob(os.path.join(results_dir, f"{m}_seed_*.csv"))
        if not csv_files: continue
        
        df_list_in = []
        df_list_sh = []
        for f in csv_files:
            df = pd.read_csv(f)
            if 'N_Points' not in df.columns: continue
            df = df.set_index('N_Points')
            df_list_in.append(df['Entropy_Input'])
            df_list_sh.append(df['Entropy_SHAP'])
            
        cmap_dict = {"LHS": "Greys", "SUR_SHAP": "Greens", "V6_SUR": "Blues", "V6_DYN": "Reds"}
        
        # Invisible scatter for the legend
        plt.scatter([], [], color=colors[m], label=labels[m], s=60)
        
        # Plot all 250 points (5 seeds * 50 iterations) with gradient
        for df_in, df_sh in zip(df_list_in, df_list_sh):
            iterations = np.linspace(0.3, 1.0, len(df_in))
            plt.scatter(df_in, df_sh, c=iterations, cmap=cmap_dict[m], s=30, edgecolors='none', zorder=3, alpha=0.6)
            
        # Re-calculate and plot median arrows
        med_in = pd.concat(df_list_in, axis=1).median(axis=1)
        med_sh = pd.concat(df_list_sh, axis=1).median(axis=1)
        
        # Calculate vectors for quiver
        X = med_in.iloc[:-1].values
        Y = med_sh.iloc[:-1].values
        U = med_in.iloc[1:].values - X
        V = med_sh.iloc[1:].values - Y
        
        # Draw arrows for the median (skip LHS as it's just random DoE)
        if m != "LHS":
            plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=colors[m], alpha=0.9, width=0.003, headwidth=5, headlength=5, edgecolors='black', linewidth=0.5, zorder=4)

    plt.xlabel('Physical Diversity (Normalized Input Entropy)')
    plt.ylabel('Explanatory Diversity (Normalized SHAP Entropy)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", "dual_space_efficiency.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    print(f"Saved dual space efficiency plot to {out_img}")

if __name__ == "__main__":
    plot_dual_space()
