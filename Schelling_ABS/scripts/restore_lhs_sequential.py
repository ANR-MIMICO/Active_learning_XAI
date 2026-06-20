import sys, os
import numpy as np
import pandas as pd
import multiprocessing as mp
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import compute_metrics
from paper_benchmark import prepare_simulator, schelling_simulator

def restore_lhs_seed(seed):
    print(f"--- RESTORING LHS - Seed {seed} ---")
    np.random.seed(seed)
    
    mlp, scaler = prepare_simulator()
    simulator = lambda x: schelling_simulator(x, mlp, scaler)
    
    design_space = DesignSpace([
        IntegerVariable(2, 5), FloatVariable(0.01, 1.0), FloatVariable(0.0, 1.0), 
        IntegerVariable(10, 40), IntegerVariable(1, 10),
    ])
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    csv_path = os.path.join(out_dir, f"LHS_seed_{seed}.csv")
    folder = os.path.join(out_dir, f"tmp_lhs_{seed}")
    os.makedirs(folder, exist_ok=True)
    
    # 1. Generate 80-point LHS
    samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
    x_full = samp(80)
    for i, var in enumerate(design_space.design_variables):
        if isinstance(var, IntegerVariable): x_full[:, i] = np.round(x_full[:, i])
    
    y_full = simulator(x_full)
    
    with open(csv_path, 'w') as f:
        f.write("Loop,N_Points,Entropy_Input,CV_NND_Input,Entropy_SHAP,CV_NND_SHAP\n")
        
    for n in range(30, 81):
        X_curr, y_curr = x_full[:n], y_full[:n]
        
        # Train Kriging
        sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space)
        sm.set_training_values(X_curr, y_curr)
        sm.train()
        explainer = shap.ExactExplainer(sm.predict_values, X_curr)
        shap_curr = explainer(X_curr).values
        ent_x, cv_x, ent_s, cv_s = compute_metrics(X_curr, shap_curr, design_space)
        
        with open(csv_path, 'a') as f:
            f.write(f"{n-30},{n},{ent_x:.4f},{cv_x:.4f},{ent_s:.4f},{cv_s:.4f}\n")
            
        # SAVE AL DATABASE LOOP N FOR GIFS!
        df = pd.DataFrame(X_curr, columns=["Var1", "Var2", "Var3", "Var4", "Var5"])
        df['y'] = y_curr
        df.to_csv(os.path.join(folder, f"al_database_loop_{n-30}.csv"), index=False)
        if n == 80:
            df.to_csv(os.path.join(folder, "al_database.csv"), index=False)

    print(f"--- FINISHED RESTORING LHS - Seed {seed} ---")
    return True

if __name__ == '__main__':
    seeds = [42, 100, 2026, 777, 12345]
    print("Launching LHS restore on 5 cores...")
    with mp.Pool(processes=5) as pool:
        pool.map(restore_lhs_seed, seeds)
    
    import plot_paper_benchmark
    plot_paper_benchmark.plot_paper_benchmark()
    import plot_all_metrics
    plot_all_metrics.plot_all_metrics()
    print("All plots restored!")
