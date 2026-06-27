import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import glob

def get_simulator():
    """Trains the MLP to get the true probability boundary (Ground Truth)"""
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'doe_5_200'))
    DATABASEx = []
    DATABASEy = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                last_line = file.readlines()[-1]
                DATABASEx.append(np.array(filename[:-11].split('_'), dtype=np.float64))
                DATABASEy.append(np.array([bool(last_line.split(',')[0].lower() == "true"), float(last_line.split(',')[1])]))
    
    DATABASEx = np.array(DATABASEx)
    DATABASEy = np.array(DATABASEy)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(DATABASEx)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_scaled, DATABASEy[:, 0].astype(int))
    return mlp, scaler

def plot_boundary():
    mlp, scaler = get_simulator()
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    
    methods = ["LHS", "SUR_SHAP", "V6_SUR", "V6_Dynamic"]
    colors = {"LHS": "black", "SUR_SHAP": "green", "V6_SUR": "blue", "V6_Dynamic": "red"}
    labels = {"LHS": "LHS", "SUR_SHAP": "SHAP-CS", "V6_SUR": "IMSE-US", "V6_Dynamic": "Dynamic-US"}
    
    plt.figure(figsize=(10, 6))
    
    for m in methods:
        # Load all points sampled by method m across all seeds
        # Actually the CSVs only save metrics, not the points themselves!
        # Wait... al_metrics_history.csv doesn't save X! We must load the al_database.csv
        # Map the formal method name back to its folder prefix
        folder_prefix = m.lower()
        if m == "V6_Dynamic":
            folder_prefix = "v6_dyn"
        
        csv_files = glob.glob(os.path.join(results_dir, f"tmp_{folder_prefix}_*", "al_database.csv"))
        if len(csv_files) == 0:
            csv_files = glob.glob(os.path.join(results_dir, f"tmp_{folder_prefix}_*", "al_database_loop_49.csv"))
            
        all_probs = []
        for db_path in csv_files:
                df = pd.read_csv(db_path)
                # X variables are the first 5 columns
                X_sampled = df.iloc[:, :5].values
                # We only want the newly added points, not the 30 LHS initial points
                X_added = X_sampled[30:]
                if len(X_added) > 0:
                    X_scaled = scaler.transform(X_added)
                    probs = mlp.predict_proba(X_scaled)[:, 1]
                    all_probs.extend(probs)
                    
        if len(all_probs) > 0:
            sns.kdeplot(all_probs, color=colors[m], label=labels[m], lw=3, fill=True, alpha=0.1)
            
    plt.axvline(0.5, color='black', linestyle='--', label="Tipping Point Boundary (P=0.5)")
    plt.title('Distribution of Evaluated Points relative to the Tipping Point', fontsize=16, fontweight='bold')
    plt.xlabel('Probability of Convergence (Simulated Output)', fontsize=14)
    plt.ylabel('Density of Sampled Points', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 1)
    
    out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", "tipping_point_distribution.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"Saved Tipping Point Boundary plot to {out_img}")

if __name__ == '__main__':
    plot_boundary()
