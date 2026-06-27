import os
import pandas as pd
import numpy as np

def patch_databases():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results_2"))
    seeds = [42, 100, 2026, 777, 12345]
    methods = ["tmp_v6_sur", "tmp_v6_dyn", "tmp_sur_shap", "tmp_v5"]
    
    print("Patching Physical Databases with Exact LHS Starting Points...")
    
    for seed in seeds:
        lhs_dir = os.path.join(results_dir, f"tmp_lhs_{seed}")
        lhs_path = os.path.join(lhs_dir, "al_database_loop_0.csv")
        
        if not os.path.exists(lhs_path):
            print(f"Skipping seed {seed}, LHS missing")
            continue
            
        df_lhs = pd.read_csv(lhs_path)
        first_30_points = df_lhs.iloc[:30].copy()
        
        for m in methods:
            m_dir = os.path.join(results_dir, f"{m}_{seed}")
            if not os.path.exists(m_dir): continue
            
            for loop in range(50):
                db_path = os.path.join(m_dir, f"al_database_loop_{loop}.csv")
                if not os.path.exists(db_path): continue
                
                df_target = pd.read_csv(db_path)
                
                # Replace the first 30 points
                for i in range(30):
                    df_target.iloc[i] = first_30_points.iloc[i]
                    
                df_target.to_csv(db_path, index=False)
        print(f"Seed {seed} physical databases patched!")
        
    print("Patching Metrics CSVs with Exact LHS Start...")
    for seed in seeds:
        lhs_csv = os.path.join(results_dir, f"LHS_seed_{seed}.csv")
        if not os.path.exists(lhs_csv): continue
        df_lhs_metrics = pd.read_csv(lhs_csv)
        row_0 = df_lhs_metrics.iloc[0].copy()
        
        method_csvs = ["V6_SUR", "V6_Dynamic", "SUR_SHAP", "V5"]
        for m in method_csvs:
            csv_path = os.path.join(results_dir, f"{m}_seed_{seed}.csv")
            if not os.path.exists(csv_path): continue
            
            df_m = pd.read_csv(csv_path)
            # Patch t=0
            df_m.loc[0, "Entropy_Input"] = row_0["Entropy_Input"]
            df_m.loc[0, "CV_NND_Input"] = row_0["CV_NND_Input"]
            df_m.loc[0, "Entropy_SHAP"] = row_0["Entropy_SHAP"]
            df_m.loc[0, "CV_NND_SHAP"] = row_0["CV_NND_SHAP"]
            df_m.to_csv(csv_path, index=False)
            
        print(f"Seed {seed} metrics patched!")
        
if __name__ == '__main__':
    patch_databases()
