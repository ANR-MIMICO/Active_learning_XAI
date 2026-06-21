import sys, os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
import shap

# Ensure imports work regardless of execution directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import ActiveLearningXAI, compute_metrics

def prepare_simulator():
    """Loads the database and trains the Schelling surrogate once."""
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'doe_5_200'))
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
    
    return mlp, scaler

def schelling_simulator(x, mlp, scaler):
    x_scaled = scaler.transform(x)
    return mlp.predict_proba(x_scaled)[:, 1]

def run_seed_task(args):
    seed, method = args
    print(f"--- STARTING: {method} - Seed {seed} ---")
    
    # 1. Setup deterministic seed for this worker
    np.random.seed(seed)
    
    # 2. Setup Simulator & Design Space
    mlp, scaler = prepare_simulator()
    simulator = lambda x: schelling_simulator(x, mlp, scaler)
    
    design_space = DesignSpace([
        IntegerVariable(2, 5), FloatVariable(0.01, 1.0), FloatVariable(0.0, 1.0), 
        IntegerVariable(10, 40), IntegerVariable(1, 10),
    ])
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{method}_seed_{seed}.csv")
    
    # 3. Generate Initial DoE (always 30 points) - Load from existing LHS to ensure identical start
    base_lhs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    lhs_csv = os.path.join(base_lhs_dir, f"tmp_lhs_{seed}", "al_database_loop_0.csv")
    
    if os.path.exists(lhs_csv):
        df_init = pd.read_csv(lhs_csv)
        x_initial = df_init.iloc[:, :5].values
        y_initial = df_init["Target"].values
    else:
        # Fallback (shouldn't happen if LHS was run)
        samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
        x_initial = samp(30)
        for i, var in enumerate(design_space.design_variables):
            if isinstance(var, IntegerVariable): x_initial[:, i] = np.round(x_initial[:, i])
        y_initial = simulator(x_initial)
    
    if method == "LHS":
        # Pure LHS from 30 to 80 points
        x_full = samp(80)
        for i, var in enumerate(design_space.design_variables):
            if isinstance(var, IntegerVariable): x_full[:, i] = np.round(x_full[:, i])
        y_full = simulator(x_full)
        
        with open(csv_path, 'w') as f:
            f.write("Loop,N_Points,Entropy_Input,CV_NND_Input,Entropy_SHAP,CV_NND_SHAP\n")
            
        for n in range(30, 81):
            X_curr, y_curr = x_full[:n], y_full[:n]
            # Train Kriging to extract SHAP
            sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space)
            sm.set_training_values(X_curr, y_curr)
            sm.train()
            explainer = shap.ExactExplainer(sm.predict_values, X_curr)
            shap_curr = explainer(X_curr).values
            ent_x, cv_x, ent_s, cv_s = compute_metrics(X_curr, shap_curr, design_space)
            
            with open(csv_path, 'a') as f:
                f.write(f"{n-30},{n},{ent_x:.4f},{cv_x:.4f},{ent_s:.4f},{cv_s:.4f}\n")
                
    elif method == "SUR":
        # Alpha fixed at 0.0
        al = ActiveLearningXAI(simulator, design_space, x_initial, y_initial, alpha_start=0.0, alpha_end=0.0, total_loops=50, mode='v4')
        al.run(output_dir=os.path.join(out_dir, f"tmp_sur_{seed}"))
        # Move the CSV
        os.rename(os.path.join(out_dir, f"tmp_sur_{seed}", "al_metrics_history.csv"), csv_path)
        
    elif method == "V4":
        # Alpha fixed at 0.5 (perfectly balanced and normalized)
        al = ActiveLearningXAI(simulator, design_space, x_initial, y_initial, alpha_start=0.5, alpha_end=0.5, total_loops=50, mode='v4')
        al.run(output_dir=os.path.join(out_dir, f"tmp_v4_{seed}"))
        # Move the CSV
        os.rename(os.path.join(out_dir, f"tmp_v4_{seed}", "al_metrics_history.csv"), csv_path)
        
    elif method == "SUR_SHAP":
        # Alpha fixed at 1.0 (Pure XAI Novelty, ignore physical uncertainty)
        al = ActiveLearningXAI(simulator, design_space, x_initial, y_initial, alpha_start=1.0, alpha_end=1.0, total_loops=50, mode='v4')
        al.run(output_dir=os.path.join(out_dir, f"tmp_sur_shap_{seed}"))
        # Move the CSV
        os.rename(os.path.join(out_dir, f"tmp_sur_shap_{seed}", "al_metrics_history.csv"), csv_path)
        
    elif method == "V5":
        # Dynamic Annealing: Starts at 0.0 (Space-US) and ends at 1.0 (SHAP-US)
        al = ActiveLearningXAI(simulator, design_space, x_initial, y_initial, alpha_start=0.0, alpha_end=1.0, total_loops=50, mode='v4')
        al.run(output_dir=os.path.join(out_dir, f"tmp_v5_{seed}"))
        # Move the CSV
        os.rename(os.path.join(out_dir, f"tmp_v5_{seed}", "al_metrics_history.csv"), csv_path)

    print(f"--- FINISHED: {method} - Seed {seed} ---")
    return True

if __name__ == '__main__':
    seeds = [42, 100, 2026, 777, 12345]
    methods = ["SUR", "V5", "SUR_SHAP"] # Run the 3 AL methods
    
    tasks = []
    for m in methods:
        for s in seeds:
            tasks.append((s, m))
            
    print(f"Launching {len(tasks)} tasks in parallel...")
    # Use CPU count minus 1 to not freeze the computer
    n_cores = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=n_cores) as pool:
        pool.map(run_seed_task, tasks)
        
    print("ALL TASKS COMPLETED SUCCESSFULLY. Ready for plotting!")
