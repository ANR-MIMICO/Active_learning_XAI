import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable
from smt.sampling_methods import LHS

from src.al_xai_optimizer import ActiveLearningXAI, compute_metrics

def main():
    print("--- BENCHMARK 10 POINTS: V4 vs SUR vs LHS ---")
    
    # 1. Load Initial Database from doe_5_200
    folder_path = os.path.join(os.path.dirname(__file__), '..', 'doe_5_200')
    DATABASEx = []
    DATABASEy = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                last_line = file.readlines()[-1]
                DATABASEx.append(np.array(filename[:-11].split('_'), dtype=np.float64))
                DATABASEy.append(np.array([
                    bool(last_line.split(',')[0].lower() == "true"), 
                    float(last_line.split(',')[1])
                ]))
                
    DATABASEx = np.array(DATABASEx)
    DATABASEy = np.array(DATABASEy)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(DATABASEx)
    
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_scaled, DATABASEy[:, 0].astype(int))
    
    def schelling_simulator(x):
        x_scaled = scaler.transform(x)
        return mlp.predict_proba(x_scaled)[:, 1]
        
    design_space = DesignSpace([
        IntegerVariable(2, 5), FloatVariable(0.01, 1.0), FloatVariable(0.0, 1.0), IntegerVariable(10, 40), IntegerVariable(1, 10),
    ])
    
    # Generate identical starting DoE for all methods (30 points)
    samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
    x_initial = samp(30)
    for i, var in enumerate(design_space.design_variables):
        if isinstance(var, IntegerVariable): x_initial[:, i] = np.round(x_initial[:, i])
    y_initial = schelling_simulator(x_initial)
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "v4_al_results")
    
    # --- METHOD 1: V4 (Cosine Novelty + Alpha 0->1) ---
    print("\n--- RUNNING V4 ---")
    # al_v4 = ActiveLearningXAI(schelling_simulator, design_space, x_initial, y_initial, alpha_start=0.0, alpha_end=1.0, total_loops=10)
    # al_v4.run(output_dir=os.path.join(out_dir, "v4_run"))
    df_v4 = pd.read_csv(os.path.join(out_dir, "v4_run", "al_metrics_history.csv"))
    
    # --- METHOD 2: SUR (Variance only, Alpha 0->0) ---
    print("\n--- RUNNING SUR ---")
    # al_sur = ActiveLearningXAI(schelling_simulator, design_space, x_initial, y_initial, alpha_start=0.0, alpha_end=0.0, total_loops=10, mode='v4')
    # al_sur.run(output_dir=os.path.join(out_dir, "sur_run"))
    df_sur = pd.read_csv(os.path.join(out_dir, "sur_run", "al_metrics_history.csv"))
    
    # --- METHOD 3: V3 (KDE 1D Soft Folding, Alpha 0->1) ---
    print("\n--- RUNNING V3 (KDE) ---")
    # al_v3 = ActiveLearningXAI(schelling_simulator, design_space, x_initial, y_initial, alpha_start=0.0, alpha_end=1.0, total_loops=10, mode='v3')
    # al_v3.run(output_dir=os.path.join(out_dir, "v3_run"))
    df_v3 = pd.read_csv(os.path.join(out_dir, "v3_run", "al_metrics_history.csv"))

    # --- Plot Comparison ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_v4['N_Points'], df_v4['Entropy_SHAP'], label='V4 (Cosinus Novelty + Uncertainty)', marker='o', lw=2, color='red')
    plt.plot(df_v3['N_Points'], df_v3['Entropy_SHAP'], label='V3 (KDE Soft Folding)', marker='^', lw=2, color='orange')
    plt.plot(df_sur['N_Points'], df_sur['Entropy_SHAP'], label='SUR (Uncertainty Only)', marker='s', lw=2, color='blue')
    
    # --- METHOD 4: Pure LHS (Baseline) ---
    # We plot LHS as a baseline. The entropy of a 40-point LHS is computed.
    x_lhs_40 = samp(40)
    for i, var in enumerate(design_space.design_variables):
        if isinstance(var, IntegerVariable): x_lhs_40[:, i] = np.round(x_lhs_40[:, i])
    # To plot LHS, we just assume its entropy would be a flat or linearly growing line, but actually we can compute it at 40
    # For simplicity of the benchmark plot, we add it as a scatter point or horizontal baseline at 40 points
    from smt.surrogate_models import KRG
    import shap
    y_lhs_40 = schelling_simulator(x_lhs_40)
    sm_lhs = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space)
    sm_lhs.set_training_values(x_lhs_40, y_lhs_40)
    sm_lhs.train()
    expl_lhs = shap.ExactExplainer(sm_lhs.predict_values, x_lhs_40)
    shap_lhs_40 = expl_lhs(x_lhs_40).values
    ent_lhs_40, _, _, _ = compute_metrics(x_lhs_40, shap_lhs_40, design_space)
    plt.scatter([40], [ent_lhs_40], color='black', s=150, zorder=5, marker='*', label='Pure LHS (40 pts)')
    
    plt.title('SHAP Entropy Comparison (4 Algorithms, 10 Loops)', fontweight='bold')
    plt.xlabel('Number of Evaluated Points')
    plt.ylabel('Normalized SHAP Entropy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    out_img = os.path.join(out_dir, "..", "..", "figures", "analysis", "benchmark_4_algos.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"\nSaved Benchmark plot to {out_img}")

if __name__ == "__main__":
    main()
