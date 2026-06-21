import sys, os
import numpy as np
import pandas as pd
from smt.design_space import DesignSpace, FloatVariable
from smt.sampling_methods import LHS
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

# Import the AL Framework
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import ActiveLearningXAI

def prepare_simulator():
    folder_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    csv_file = os.path.join(folder_path, 'Newbounds.csv')
    
    df = pd.read_csv(csv_file)
    df["density"] = np.log10(df["density"])
    df["scarcity"] = np.log2(df["scarcity"])
    
    DATABASEx = np.array(df)[:,:5]
    DATABASEy = np.array(df)[:,5] # PRICE
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(DATABASEx)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X_scaled, DATABASEy)
    
    def simulator(x):
        x_scaled = scaler.transform(x)
        return mlp.predict(x_scaled)
        
    return simulator

def run_seed_task(args):
    seed, method = args
    print(f"--- STARTING: {method} - Seed {seed} ---")
    np.random.seed(seed)
    
    simulator = prepare_simulator()
    
    design_space = DesignSpace([
        FloatVariable(0, 200),  # 0. price_to_dispose
        FloatVariable(-2, 2),   # 1. scarcity (log2)
        FloatVariable(-4, -1),  # 2. density (log10)
        FloatVariable(0, 0.5),  # 3. cluster_spread
        FloatVariable(0, 10),   # 4. km_cost
    ])
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{method}_seed_{seed}.csv")
    
    # Generate or load 30 points
    lhs_csv = os.path.join(out_dir, f"tmp_lhs_{seed}", "al_database_loop_0.csv")
    
    if method != "LHS" and os.path.exists(lhs_csv):
        df_init = pd.read_csv(lhs_csv)
        x_initial = df_init.iloc[:, :5].values
        y_initial = df_init["Target"].values
    else:
        samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
        x_initial = samp(30)
        y_initial = simulator(x_initial)
    
    if method == "LHS":
        samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
        x_full = samp(80)
        y_full = simulator(x_full)
        
        tmp_dir = os.path.join(out_dir, f"tmp_lhs_{seed}")
        os.makedirs(tmp_dir, exist_ok=True)
        
        df_init = pd.DataFrame(x_full[:30], columns=[f"Var_{i}" for i in range(5)])
        df_init["Target"] = y_full[:30]
        df_init.to_csv(os.path.join(tmp_dir, "al_database_loop_0.csv"), index=False)
        
        # We don't necessarily generate full metrics_history for pure LHS unless we rewrite compute_metrics
        # Just generate the final dataframe
        df_combined = pd.DataFrame(x_full, columns=[f"Var_{i}" for i in range(5)])
        df_combined["Target"] = y_full
        df_combined.to_csv(csv_path, index=False)
        
    elif method == "SUR":
        al = ActiveLearningXAI(simulator, design_space, x_initial, y_initial, alpha_start=0.0, alpha_end=0.0, total_loops=50, mode='v4')
        al.run(output_dir=os.path.join(out_dir, f"tmp_sur_{seed}"))
        os.rename(os.path.join(out_dir, f"tmp_sur_{seed}", "al_metrics_history.csv"), csv_path)
        
    elif method == "SUR_SHAP":
        al = ActiveLearningXAI(simulator, design_space, x_initial, y_initial, alpha_start=1.0, alpha_end=1.0, total_loops=50, mode='v4')
        al.run(output_dir=os.path.join(out_dir, f"tmp_sur_shap_{seed}"))
        os.rename(os.path.join(out_dir, f"tmp_sur_shap_{seed}", "al_metrics_history.csv"), csv_path)
        
    elif method == "V5":
        al = ActiveLearningXAI(simulator, design_space, x_initial, y_initial, alpha_start=0.0, alpha_end=1.0, total_loops=50, mode='v4')
        al.run(output_dir=os.path.join(out_dir, f"tmp_v5_{seed}"))
        os.rename(os.path.join(out_dir, f"tmp_v5_{seed}", "al_metrics_history.csv"), csv_path)

    print(f"--- FINISHED: {method} - Seed {seed} ---")
    return True

if __name__ == '__main__':
    # First run LHS sequentially to generate the base 30 points
    seeds = [42, 100, 2026, 777, 12345]
    print("Generating LHS baselines...")
    for s in seeds:
        run_seed_task((s, "LHS"))
        
    print("Launching AL algorithms in parallel...")
    methods = ["SUR", "V5", "SUR_SHAP"]
    
    tasks = []
    for m in methods:
        for s in seeds:
            tasks.append((s, m))
            
    n_cores = min(len(tasks), mp.cpu_count() - 1)
    if n_cores < 1: n_cores = 1
    
    with mp.Pool(n_cores) as pool:
        pool.map(run_seed_task, tasks)
    
    print("ALL TASKS COMPLETED SUCCESSFULLY.")
