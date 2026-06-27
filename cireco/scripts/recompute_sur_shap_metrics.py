import os, glob
import pandas as pd
import numpy as np
import shap
from smt.surrogate_models import KRG
from smt.design_space import DesignSpace, FloatVariable
import sys
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import compute_metrics

def recompute_seed(seed):
    print(f"Recomputing SUR_SHAP for seed {seed}")
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    method_dir = os.path.join(results_dir, f"tmp_sur_shap_{seed}")
    
    design_space = DesignSpace([
        FloatVariable(0, 200),  
        FloatVariable(-2, 2),   
        FloatVariable(-4, -1),  
        FloatVariable(0, 0.5),  
        FloatVariable(0, 10),   
    ])
    
    metrics_history = []
    
    for loop in range(50):
        db_path = os.path.join(method_dir, f"al_database_loop_{loop}.csv")
        if not os.path.exists(db_path): break
        
        df = pd.read_csv(db_path)
        X = df.iloc[:, :5].values
        Y = df["Target"].values
        
        sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space)
        sm.set_training_values(X, Y)
        sm.train()
        
        explainer = shap.ExactExplainer(sm.predict_values, X)
        shap_values = explainer(X).values
        
        ent_x, cv_x, ent_shap, cv_shap = compute_metrics(X, shap_values, design_space)
        
        metrics_history.append({
            "Loop": loop,
            "N_Points": 30 + loop,
            "Alpha": 1.0,
            "Entropy_Input": ent_x,
            "CV_NND_Input": cv_x,
            "Entropy_SHAP": ent_shap,
            "CV_NND_SHAP": cv_shap
        })
        
    out_df = pd.DataFrame(metrics_history)
    out_df.to_csv(os.path.join(results_dir, f"SUR_SHAP_seed_{seed}.csv"), index=False)
    print(f"Seed {seed} done!")

if __name__ == "__main__":
    seeds = [42, 100, 2026, 777, 12345]
    with mp.Pool(5) as pool:
        pool.map(recompute_seed, seeds)
