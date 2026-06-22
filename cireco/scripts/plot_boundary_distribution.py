import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import glob


def plot_boundary():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    
    methods = ["LHS", "SUR_SHAP", "V6_SUR", "V6_DYN"]
    colors = {"LHS": "black", "SUR_SHAP": "green", "V6_SUR": "blue", "V6_DYN": "red"}
    labels = {"LHS": "LHS", "SUR_SHAP": "SHAP-CS", "V6_SUR": "IMSE-US", "V6_DYN": "Dynamic-US"}
    
    plt.figure(figsize=(10, 6))
    
    for m in methods:
        # Load all points sampled by method m across all seeds
        # Actually the CSVs only save metrics, not the points themselves!
        # Wait... al_metrics_history.csv doesn't save X! We must load the al_database.csv
        csv_files = glob.glob(os.path.join(results_dir, f"tmp_{m.lower()}_*", "al_database.csv"))
        
        # If the folders were renamed, we need to adapt. Since we renamed the CSVs and deleted the folders, we lost the points!
        # Wait, in paper_benchmark, I only renamed the CSV and left the folders `tmp_v5_seed` intact?
        csv_files = glob.glob(os.path.join(results_dir, f"tmp_{m.lower()}_*", "al_database.csv"))
        if len(csv_files) == 0:
            csv_files = glob.glob(os.path.join(results_dir, f"tmp_{m.lower()}_*", "al_database_loop_49.csv"))
            
        all_probs = []
        for db_path in csv_files:
                df = pd.read_csv(db_path)
                # X variables are the first 5 columns
                X_sampled = df.iloc[:, :5].values
                # We only want the newly added points, not the 30 LHS initial points
                X_added = X_sampled[30:]
                if len(X_added) > 0:
                    probs = df["Target"].values[30:]
                    all_probs.extend(probs)
                    
        if len(all_probs) > 0:
            sns.kdeplot(all_probs, color=colors[m], label=labels[m], lw=3, fill=True, alpha=0.1)
            
    plt.axvline(70, color='black', linestyle='--', label="Price Threshold")
    plt.title('Distribution of Evaluated Points relative to the Tipping Point', fontsize=16, fontweight='bold')
    plt.xlabel('Simulated Price (Target)', fontsize=14)
    plt.ylabel('Density of Sampled Points', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    
    out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", "tipping_point_distribution.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"Saved Tipping Point Boundary plot to {out_img}")

if __name__ == '__main__':
    plot_boundary()
