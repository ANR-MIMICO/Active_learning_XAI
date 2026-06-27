import os
import pandas as pd
import shutil

results_dir = r"C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\data\processed\paper_results_2"
orig_lhs_dir = r"C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\data\processed\paper_results"

seeds = [42, 100, 2026, 777, 12345]
methods_to_unpatch = ["V6_SUR", "V6_DYN", "V6_Dynamic"] # check both namings

for seed in seeds:
    # Get true first 30 points from orig LHS
    orig_lhs_csv = os.path.join(orig_lhs_dir, f"tmp_lhs_{seed}", "al_database_loop_0.csv")
    if not os.path.exists(orig_lhs_csv): continue
    df_true_start = pd.read_csv(orig_lhs_csv)
    
    # Get true metrics at loop 1 (or loop 0) from the newly generated SUR_SHAP
    sur_shap_csv = os.path.join(results_dir, f"SUR_SHAP_seed_{seed}.csv")
    if not os.path.exists(sur_shap_csv): continue
    df_sur_shap = pd.read_csv(sur_shap_csv)
    true_row_0 = df_sur_shap.iloc[0] # This contains the true metrics for df_true_start
    
    for method in methods_to_unpatch:
        # Unpatch metrics history
        metrics_csv = os.path.join(results_dir, f"{method}_seed_{seed}.csv")
        if os.path.exists(metrics_csv):
            df_metrics = pd.read_csv(metrics_csv)
            # Restore the metrics
            df_metrics.loc[0, "Entropy_Input"] = true_row_0["Entropy_Input"]
            df_metrics.loc[0, "CV_NND_Input"] = true_row_0["CV_NND_Input"]
            df_metrics.loc[0, "Entropy_SHAP"] = true_row_0["Entropy_SHAP"]
            df_metrics.loc[0, "CV_NND_SHAP"] = true_row_0["CV_NND_SHAP"]
            df_metrics.to_csv(metrics_csv, index=False)
            print(f"Unpatched metrics for {method} seed {seed}")

        # Unpatch temporary databases
        tmp_dir = os.path.join(results_dir, f"tmp_{method.lower()}_{seed}")
        if not os.path.exists(tmp_dir) and method == "V6_Dynamic":
            tmp_dir = os.path.join(results_dir, f"tmp_v6_dyn_{seed}")
            
        if os.path.exists(tmp_dir):
            for i in range(50):
                db_csv = os.path.join(tmp_dir, f"al_database_loop_{i}.csv")
                if os.path.exists(db_csv):
                    df_db = pd.read_csv(db_csv)
                    # Restore the true first 30 points
                    for j in range(30):
                        df_db.iloc[j] = df_true_start.iloc[j]
                    df_db.to_csv(db_csv, index=False)
            print(f"Unpatched DBs for {method} seed {seed}")
