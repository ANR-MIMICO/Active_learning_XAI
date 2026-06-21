import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def get_data_for_seed(seed=42):
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    
    # LHS is raw csv, others are in tmp_ folders
    df_lhs = pd.read_csv(os.path.join(results_dir, f"LHS_seed_{seed}.csv"))
    df_sur = pd.read_csv(os.path.join(results_dir, f"tmp_sur_{seed}", "al_database_loop_49.csv"))
    df_v5 = pd.read_csv(os.path.join(results_dir, f"tmp_v5_{seed}", "al_database_loop_49.csv"))
    df_sur_shap = pd.read_csv(os.path.join(results_dir, f"tmp_sur_shap_{seed}", "al_database_loop_49.csv"))
    
    return df_lhs, df_sur, df_sur_shap, df_v5

def plot_target_distribution():
    print("Generating Target (Price) Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_lhs, df_sur, df_sur_shap, df_v5 = get_data_for_seed(42)
    
    # We only want the ADDED points for AL methods
    prices_lhs = df_lhs['Target'].values
    prices_sur = df_sur['Target'].values[30:]
    prices_v5 = df_v5['Target'].values[30:]
    prices_shap = df_sur_shap['Target'].values[30:]
    
    sns.kdeplot(prices_lhs, label="LHS (Uniform)", color="black", lw=2, fill=True, alpha=0.1, ax=ax)
    sns.kdeplot(prices_sur, label="Space-US", color="blue", lw=2, fill=True, alpha=0.1, ax=ax)
    sns.kdeplot(prices_v5, label="Dynamic-US", color="red", lw=2, fill=True, alpha=0.1, ax=ax)
    sns.kdeplot(prices_shap, label="SHAP-US", color="green", lw=2, fill=True, alpha=0.1, ax=ax)
    
    ax.set_title("Distribution of Explored Targets (Price)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Price", fontsize=14)
    ax.set_ylabel("Density of Sampled Points", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    out_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "analysis", "target_distribution.png"))
    plt.savefig(out_img, dpi=300)
    plt.close()

def plot_dual_space():
    print("Generating Dual Space Efficiency...")
    df_lhs, df_sur, df_sur_shap, df_v5 = get_data_for_seed(42)
    
    # Calculate diversity
    from scipy.stats import entropy
    def get_entropy(X):
        n_bins=10
        ent_list = []
        for i in range(X.shape[1]):
            hist, _ = np.histogram(X[:, i], bins=n_bins, density=False)
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]
            ent_list.append(entropy(hist) / np.log(n_bins))
        return np.mean(ent_list)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get the metrics CSV for the final scores
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    methods = [("SUR", "blue", "Space-US"), ("V5", "red", "Dynamic-US"), ("SUR_SHAP", "green", "SHAP-US")]
    
    for m, col, label in methods:
        f = os.path.join(results_dir, f"{m}_seed_42.csv")
        df = pd.read_csv(f)
        x_val = df['Entropy_Input'].iloc[-1]
        y_val = df['Entropy_SHAP'].iloc[-1]
        ax.scatter(x_val, y_val, color=col, s=200, label=label, edgecolor='black', zorder=5)
        
    ax.set_xlabel("Physical Diversity (Input Entropy)", fontsize=14)
    ax.set_ylabel("Explanatory Diversity (SHAP Entropy)", fontsize=14)
    ax.set_title("Dual Space Efficiency (Final State)", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    
    out_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "analysis", "dual_space_efficiency.png"))
    plt.savefig(out_img, dpi=300)
    plt.close()

def plot_pca():
    print("Generating PCA Scatter...")
    df_lhs, df_sur, df_sur_shap, df_v5 = get_data_for_seed(42)
    X_lhs = df_lhs.iloc[:, :5].values
    X_v5 = df_v5.iloc[:, :5].values
    
    pca = PCA(n_components=2)
    pca.fit(X_lhs) # Fit on baseline
    
    pca_lhs = pca.transform(X_lhs)
    pca_v5 = pca.transform(X_v5[30:]) # Only new points
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_lhs[:, 0], pca_lhs[:, 1], c='gray', alpha=0.3, label='LHS (Baseline)', s=50)
    ax.scatter(pca_v5[:, 0], pca_v5[:, 1], c='red', alpha=0.8, label='Dynamic-US (Added)', s=80, edgecolor='black')
    
    ax.set_title("PCA Projection of the Sampled Space", fontsize=16, fontweight='bold')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=14)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    out_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "analysis", "pca_scatter.png"))
    plt.savefig(out_img, dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_target_distribution()
    plot_dual_space()
    plot_pca()
    print("All additional plots generated!")
