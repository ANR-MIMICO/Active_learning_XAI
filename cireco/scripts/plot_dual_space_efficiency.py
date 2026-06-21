import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_dual_space():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    
    methods = ["LHS", "SUR", "SUR_SHAP", "V5"]
    colors = {"LHS": "black", "SUR": "blue", "SUR_SHAP": "green", "V5": "red"}
    labels = {
        "LHS": "LHS", 
        "SUR": "Space-US", 
        "SUR_SHAP": "SHAP-US",
        "V5": "Dynamic-US"
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
            
        med_in = pd.concat(df_list_in, axis=1).median(axis=1)
        med_sh = pd.concat(df_list_sh, axis=1).median(axis=1)
        
        # Plot points and lines lightly
        plt.plot(med_in, med_sh, '-o', label=labels[m], color=colors[m], markersize=4, lw=1, alpha=0.4)
        
        # Calculate vectors for quiver
        X = med_in.iloc[:-1].values
        Y = med_sh.iloc[:-1].values
        U = med_in.iloc[1:].values - X
        V = med_sh.iloc[1:].values - Y
        
        # Draw perfect arrows between points using quiver
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=colors[m], alpha=0.7, width=0.003, headwidth=5, headlength=5)
        
        # Highlight start and end points
        plt.scatter(med_in.iloc[0], med_sh.iloc[0], color='grey', marker='*', s=200, zorder=5)
        plt.scatter(med_in.iloc[-1], med_sh.iloc[-1], color=colors[m], marker='X', s=150, zorder=5)

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
