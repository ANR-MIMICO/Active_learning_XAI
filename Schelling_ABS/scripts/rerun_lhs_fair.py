import sys, os
import numpy as np
import pandas as pd
import multiprocessing as mp
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
import shap

# Ensure imports work regardless of execution directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import compute_metrics
from paper_benchmark import prepare_simulator, schelling_simulator

def run_fair_lhs_seed(seed):
    print(f"--- STARTING FAIR LHS - Seed {seed} ---")
    np.random.seed(seed)
    
    mlp, scaler = prepare_simulator()
    simulator = lambda x: schelling_simulator(x, mlp, scaler)
    
    design_space = DesignSpace([
        IntegerVariable(2, 5), FloatVariable(0.01, 1.0), FloatVariable(0.0, 1.0), 
        IntegerVariable(10, 40), IntegerVariable(1, 10),
    ])
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    csv_path = os.path.join(out_dir, f"LHS_seed_{seed}.csv")
    
    with open(csv_path, 'w') as f:
        f.write("Loop,N_Points,Entropy_Input,CV_NND_Input,Entropy_SHAP,CV_NND_SHAP\n")
        
    for n in range(30, 81):
        # FRESH LHS FOR EVERY N !
        # We need a new random state so it doesn't give the same thing? 
        # Actually, if we just call samp(n), it generates a fresh n-point LHS
        samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
        X_curr = samp(n)
        for i, var in enumerate(design_space.design_variables):
            if isinstance(var, IntegerVariable): X_curr[:, i] = np.round(X_curr[:, i])
        y_curr = simulator(X_curr)
        
        # Train Kriging
        sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space)
        sm.set_training_values(X_curr, y_curr)
        sm.train()
        explainer = shap.ExactExplainer(sm.predict_values, X_curr)
        shap_curr = explainer(X_curr).values
        ent_x, cv_x, ent_s, cv_s = compute_metrics(X_curr, shap_curr, design_space)
        
        with open(csv_path, 'a') as f:
            f.write(f"{n-30},{n},{ent_x:.4f},{cv_x:.4f},{ent_s:.4f},{cv_s:.4f}\n")
            
        # We also need to save al_database.csv ONLY for the boundary distribution script.
        # It expects al_database.csv in tmp_lhs_{seed} for n=80
        if n == 80:
            folder = os.path.join(out_dir, f"tmp_lhs_{seed}")
            os.makedirs(folder, exist_ok=True)
            df = pd.DataFrame(X_curr, columns=["Var1", "Var2", "Var3", "Var4", "Var5"])
            df['y'] = 0.5
            df.to_csv(os.path.join(folder, "al_database.csv"), index=False)

    print(f"--- FINISHED FAIR LHS - Seed {seed} ---")
    return True

if __name__ == '__main__':
    seeds = [42, 100, 2026, 777, 12345]
    print("Launching FAIR LHS recalculation on 5 cores...")
    with mp.Pool(processes=5) as pool:
        pool.map(run_fair_lhs_seed, seeds)
    
    # Rerun the plots automatically!
    import plot_paper_benchmark
    plot_paper_benchmark.plot_paper_benchmark()
    import plot_all_metrics
    plot_all_metrics.plot_all_metrics()
    print("All plots updated with FAIR LHS!")
