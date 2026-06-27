import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from smt.surrogate_models import KRG
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import compute_metrics
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable

design_space = DesignSpace([
    FloatVariable(0, 1),
    FloatVariable(0, 1),
    IntegerVariable(20, 80),
    IntegerVariable(3, 8),
    FloatVariable(0, 1)
])

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))

def recompute_seed(seed):
    print(f"Recomputing LHS metrics for seed {seed}...")
    lhs_dir = os.path.join(base_dir, f"tmp_lhs_{seed}")
    if not os.path.exists(lhs_dir):
        print(f"Warning: {lhs_dir} not found!")
        return

    csv_path = os.path.join(out_dir, f"LHS_seed_{seed}.csv")
    with open(csv_path, 'w') as f:
        f.write("Loop,N_Points,Entropy_Input,CV_NND_Input,Entropy_SHAP,CV_NND_SHAP\n")

    for loop in range(51):
        db_path = os.path.join(lhs_dir, f"al_database_loop_{loop}.csv")
        if not os.path.exists(db_path):
            print(f"Finished at loop {loop-1} for seed {seed}")
            break
        
        df = pd.read_csv(db_path)
        X_curr = df.iloc[:, :5].values
        y_curr = df.iloc[:, -1].values
        
        sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space)
        sm.set_training_values(X_curr, y_curr)
        sm.train()
        
        explainer = shap.ExactExplainer(sm.predict_values, X_curr)
        shap_curr = explainer(X_curr).values
        
        ent_x, cv_x, ent_s, cv_s = compute_metrics(X_curr, shap_curr, design_space)
        
        with open(csv_path, 'a') as f:
            f.write(f"{loop},{len(X_curr)},{ent_x:.4f},{cv_x:.4f},{ent_s:.4f},{cv_s:.4f}\n")
            
    print(f"Seed {seed} complete!")

if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    seeds = [42, 100, 2026, 777, 12345]
    with mp.Pool(processes=5) as pool:
        pool.map(recompute_seed, seeds)
    print("All LHS metrics successfully recomputed and saved to paper_results_2!")
