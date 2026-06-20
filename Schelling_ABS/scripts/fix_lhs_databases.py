import os, sys
import numpy as np
import pandas as pd
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable
from smt.sampling_methods import LHS

# Ensure imports work regardless of execution directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import ActiveLearningXAI

def fix_lhs():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    seeds = [42, 100, 2026, 777, 12345]
    
    design_space = DesignSpace([
        IntegerVariable(2, 5), FloatVariable(0.01, 1.0), FloatVariable(0.0, 1.0), 
        IntegerVariable(10, 40), IntegerVariable(1, 10),
    ])
    
    for seed in seeds:
        np.random.seed(seed)
        samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
        x_full = samp(80)
        for i, var in enumerate(design_space.design_variables):
            if isinstance(var, IntegerVariable): x_full[:, i] = np.round(x_full[:, i])
            
        # We need to save this to tmp_lhs_{seed}/al_database.csv
        # The al_database.csv format usually has columns for the 5 vars, plus the output 'y'.
        # Since plot_boundary_distribution.py only reads df.iloc[:, :5], we just need to save the 5 vars!
        # Let's save a dummy 'y' column just in case.
        
        folder = os.path.join(out_dir, f"tmp_lhs_{seed}")
        os.makedirs(folder, exist_ok=True)
        
        df = pd.DataFrame(x_full, columns=["Var1", "Var2", "Var3", "Var4", "Var5"])
        df['y'] = 0.5 # dummy
        df.to_csv(os.path.join(folder, "al_database.csv"), index=False)
        print(f"Created LHS database for seed {seed}")

if __name__ == '__main__':
    fix_lhs()
