import os
import pandas as pd
import numpy as np

def analyze():
    res1 = r'C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\data\processed\paper_results'
    res2 = r'C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\data\processed\paper_results_2'
    seeds = [42, 100, 777, 2026, 12345]
    
    methods = [
        ('V4 (Static)', res1, 'v4'),
        ('V5 (Dynamic)', res2, 'v5'),
        ('SUR (Space)', res2, 'sur'),
        ('SUR_SHAP', res2, 'sur_shap'),
        ('LHS', res1, 'lhs')
    ]
    
    print("Metrics at Final Loop (Mean across 5 seeds):")
    print(f"{'Method':<15} | {'Entropy_SHAP':<12} | {'CV_NND_SHAP':<12} | {'Entropy_X':<12}")
    print("-" * 60)
    
    for label, base_dir, meth in methods:
        ent_shap, cv_shap, ent_x = [], [], []
        meth_upper = meth.upper()
        for s in seeds:
            f = os.path.join(base_dir, f'{meth_upper}_seed_{s}.csv')
            if os.path.exists(f):
                df = pd.read_csv(f)
                ent_shap.append(df['Entropy_SHAP'].iloc[-1])
                cv_shap.append(df['CV_NND_SHAP'].iloc[-1])
                ent_x.append(df['Entropy_Input'].iloc[-1])
            else:
                # Fallback to tmp dir
                f_tmp = os.path.join(base_dir, f'tmp_{meth}_{s}', 'al_metrics_history.csv')
                if os.path.exists(f_tmp):
                    df = pd.read_csv(f_tmp)
                    ent_shap.append(df['Entropy_SHAP'].iloc[-1])
                    cv_shap.append(df['CV_NND_SHAP'].iloc[-1])
                    ent_x.append(df['Entropy_Input'].iloc[-1])
                    
        if ent_shap:
            print(f"{label:<15} | {np.mean(ent_shap):.4f}       | {np.mean(cv_shap):.4f}       | {np.mean(ent_x):.4f}")

if __name__ == '__main__':
    analyze()
