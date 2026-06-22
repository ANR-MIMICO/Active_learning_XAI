import sys, os
import numpy as np
import pandas as pd
from smt.design_space import DesignSpace, FloatVariable
from smt.sampling_methods import LHS
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import ActiveLearningXAI
from cireco.scripts.cireco_paper_benchmark import prepare_simulator

def run_lhs_seed(seed):
    print(f"--- STARTING LHS METRICS FOR SEED {seed} ---")
    np.random.seed(seed)
    simulator = prepare_simulator()
    
    design_space = DesignSpace([
        FloatVariable(0, 200),  
        FloatVariable(-2, 2),   
        FloatVariable(-4, -1),  
        FloatVariable(0, 0.5),  
        FloatVariable(0, 10),   
    ])
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    csv_path = os.path.join(out_dir, f"LHS_seed_{seed}.csv")
    
    metrics_history = []
    
    for n_pts in range(30, 81):
        # Different LHS distribution for each size like in the original paper!
        np.random.seed(seed * n_pts)
        samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
        x_doe = samp(n_pts)
        y_doe = simulator(x_doe)
        
        try:
            from smt.surrogate_models import KRG
            import shap
            from src.al_xai_optimizer import compute_metrics
            
            sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space)
            sm.set_training_values(x_doe, y_doe)
            sm.train()
            
            explainer = shap.ExactExplainer(sm.predict_values, x_doe)
            shap_values = explainer(x_doe).values
            
            ent_x, cvnnd_x, ent_shap, cvnnd_shap = compute_metrics(x_doe, shap_values, design_space)
            
            # Save the raw database for 1D histograms and Difference histograms
            loop_idx = n_pts - 30
            lhs_dir = os.path.join(out_dir, f"tmp_lhs_{seed}")
            os.makedirs(lhs_dir, exist_ok=True)
            df_doe = pd.DataFrame(x_doe, columns=[f"Var_{i}" for i in range(5)])
            df_doe["Target"] = y_doe
            df_doe.to_csv(os.path.join(lhs_dir, f"al_database_loop_{loop_idx}.csv"), index=False)
            
            metrics_history.append({
                "N_Points": n_pts,
                "Entropy_Input": ent_x,
                "CV_NND_Input": cvnnd_x,
                "Entropy_SHAP": ent_shap,
                "CV_NND_SHAP": cvnnd_shap
            })
        except Exception as e:
            print(f"Seed {seed} N={n_pts} MFK failed: {e}")
            continue
            
    df_metrics = pd.DataFrame(metrics_history)
    df_metrics.to_csv(csv_path, index=False)
    print(f"--- FINISHED LHS METRICS FOR SEED {seed} ---")

if __name__ == '__main__':
    # Limit OpenBLAS/OMP threads to prevent deadlock when using multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    seeds = [42, 100, 2026, 777, 12345]
    n_cores = min(len(seeds), mp.cpu_count())  # Run 1 process per seed (5 processes in parallel)
    with mp.Pool(n_cores) as pool:
        pool.map(run_lhs_seed, seeds)
