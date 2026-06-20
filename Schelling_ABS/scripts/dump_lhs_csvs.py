import os
import numpy as np
import pandas as pd
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable
from smt.sampling_methods import LHS

def dump_lhs():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    seeds = [42, 100, 2026, 777, 12345]
    
    design_space = DesignSpace([
        IntegerVariable(2, 5), FloatVariable(0.01, 1.0), FloatVariable(0.0, 1.0), 
        IntegerVariable(10, 40), IntegerVariable(1, 10),
    ])
    
    for seed in seeds:
        np.random.seed(seed)
        folder = os.path.join(out_dir, f"tmp_lhs_{seed}")
        os.makedirs(folder, exist_ok=True)
        
        for n in range(30, 81):
            samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
            X_curr = samp(n)
            for i, var in enumerate(design_space.design_variables):
                if isinstance(var, IntegerVariable): X_curr[:, i] = np.round(X_curr[:, i])
                
            df = pd.DataFrame(X_curr, columns=["Var1", "Var2", "Var3", "Var4", "Var5"])
            df['y'] = 0.5 # dummy
            loop = n - 30
            # Save exactly like ActiveLearningXAI does
            df.to_csv(os.path.join(folder, f"al_database_loop_{loop}.csv"), index=False)
            
            if n == 80:
                df.to_csv(os.path.join(folder, "al_database.csv"), index=False)

if __name__ == "__main__":
    dump_lhs()
    print("Successfully dumped 250 LHS CSV files for GIF generation!")
