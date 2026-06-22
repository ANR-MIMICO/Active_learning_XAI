import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_all_metrics():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    
    methods = ["LHS", "SUR_SHAP", "V6_SUR", "V6_DYN"]
    colors = {"LHS": "black", "SUR_SHAP": "green", "V6_SUR": "blue", "V6_DYN": "red"}
    labels = {
        "LHS": "LHS", 
        "SUR_SHAP": "SHAP-US",
        "V6_SUR": "IMSE-US",
        "V6_DYN": "Dynamic-US"
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Evolution of Metrics across 50 Iterations (Median & IQR over 5 Seeds)', fontsize=16, fontweight='bold')

    metrics = [
        ('Entropy_SHAP', 'Normalized SHAP Entropy (Higher is better)', axes[0, 0]),
        ('CV_NND_SHAP', 'CV NND SHAP (Lower is better, detects clusters)', axes[0, 1]),
        ('Entropy_Input', 'Normalized Input Entropy (Higher is better)', axes[1, 0]),
        ('CV_NND_Input', 'CV NND Input (Lower is better, detects clusters)', axes[1, 1])
    ]

    for metric_col, title, ax in metrics:
        for m in methods:
            csv_files = glob.glob(os.path.join(results_dir, f"{m}_seed_*.csv"))
            if not csv_files: continue
            
            dfs = []
            for f in csv_files:
                df = pd.read_csv(f)
                if 'N_Points' not in df.columns or metric_col not in df.columns: continue
                df = df.set_index('N_Points')
                dfs.append(df[metric_col])
            if not dfs: continue
                
            all_seeds = pd.concat(dfs, axis=1)
            median = all_seeds.median(axis=1)
            q25 = all_seeds.quantile(0.25, axis=1)
            q75 = all_seeds.quantile(0.75, axis=1)
            n_points = median.index
            
            ax.plot(n_points, median, label=labels[m], color=colors[m], lw=2)
            ax.fill_between(n_points, q25, q75, color=colors[m], alpha=0.15)
            
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Number of Evaluated Points')
        ax.set_ylabel('Metric Value')
        ax.grid(True, linestyle='--', alpha=0.6)
        if ax == axes[0,0]:
            ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", "all_metrics_evolution.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"Saved comprehensive metrics plot to {out_img}")

if __name__ == "__main__":
    plot_all_metrics()
