import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from smt.surrogate_models import KRG
import shap
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import compute_metrics
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable

# THE CORRECT DESIGN SPACE FOR SCHELLING!!!
design_space = DesignSpace([
    IntegerVariable(2, 5), 
    FloatVariable(0.01, 1.0), 
    FloatVariable(0.0, 1.0), 
    IntegerVariable(10, 40), 
    IntegerVariable(1, 10)
])

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))

def resume_seed(seed):
    print(f"Resuming LHS metrics for seed {seed} from loop 21 to 50...")
    lhs_dir = os.path.join(base_dir, f"tmp_lhs_{seed}")
    csv_path = os.path.join(out_dir, f"LHS_seed_{seed}.csv")
    
    df_existing = pd.read_csv(csv_path)
    last_loop = int(df_existing['Loop'].max())
    
    if last_loop >= 50:
        print(f"Seed {seed} already complete up to loop {last_loop}.")
        return
        
    start_loop = last_loop + 1
    
    for loop in range(start_loop, 51):
        t0 = time.time()
        db_path = os.path.join(lhs_dir, f"al_database_loop_{loop}.csv")
        if not os.path.exists(db_path):
            print(f"Missing {db_path}, stopping seed {seed}")
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
            
        print(f"Seed {seed} - Loop {loop}/50 ({len(X_curr)} pts) done in {time.time()-t0:.1f}s")
        
    print(f"Seed {seed} complete!")

if __name__ == "__main__":
    seeds = [42, 100, 2026, 777, 12345]
    with mp.Pool(processes=5) as pool:
        pool.map(resume_seed, seeds)
    print("ALL RESUMED SCHELLING METRICS SAVED!")
