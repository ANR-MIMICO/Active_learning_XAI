import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_paper_benchmark():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    
    methods = ["LHS", "SUR", "SUR_SHAP", "V6_Dynamic", "V6_SUR", "V6_Dynamic"]
    colors = {"LHS": "black", "SUR": "blue", "SUR_SHAP": "green", "V6_Dynamic": "red", "V6_SUR": "orange", "V6_Dynamic": "purple"}
    labels = {
        "LHS": "Pure LHS (Baseline)", 
        "SUR": "SUR (Uncertainty only, α=0.0)", 
        "SUR_SHAP": "SUR SHAP (Novelty only, α=1.0)",
        "V6_Dynamic": "V4 Hybrid (MFK, α=0.5)"
    }
    
    # We will aggregate data over the 5 seeds
    # Format: df_agg[method] = DataFrame grouped by N_points
    
    plt.figure(figsize=(12, 8))
    
    for m in methods:
        csv_files = glob.glob(os.path.join(results_dir, f"{m}_seed_*.csv"))
        if len(csv_files) == 0:
            print(f"No files found for method {m}")
            continue
            
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            # LHS script writes 30..80, AL script might write 30..80. We index by N_Points
            df = df.set_index('N_Points')
            dfs.append(df['Entropy_SHAP'])
            
        # Concat along columns
        all_seeds = pd.concat(dfs, axis=1)
        
        # Calculate median and percentiles
        median = all_seeds.median(axis=1)
        q25 = all_seeds.quantile(0.25, axis=1)
        q75 = all_seeds.quantile(0.75, axis=1)
        
        n_points = median.index
        
        plt.plot(n_points, median, label=labels[m], color=colors[m], lw=3)
        plt.fill_between(n_points, q25, q75, color=colors[m], alpha=0.2)
        
    plt.title('SHAP Entropy Evolution (Median & IQR over 5 Seeds)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Evaluated Points (Initial DoE=30)', fontsize=14)
    plt.ylabel('Normalized SHAP Entropy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", "paper_final_benchmark_50loops.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"Saved definitive paper plot to {out_img}")

if __name__ == '__main__':
    plot_paper_benchmark()
