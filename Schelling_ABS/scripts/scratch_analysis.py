import pandas as pd
import numpy as np
import glob
import os

res_dir = r'C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\data\processed\paper_results'
methods = ['LHS', 'SUR', 'SUR_SHAP', 'V4']

print('FINAL METRICS AT N=80 (MEDIAN ACROSS 5 SEEDS):')
for m in methods:
    fs = glob.glob(os.path.join(res_dir, f'{m}_seed_*.csv'))
    if not fs: continue
    dfs = [pd.read_csv(f).set_index('N_Points').iloc[-1] for f in fs]
    med = pd.DataFrame(dfs).median()
    print(f"{m}: Ent_SHAP={med['Entropy_SHAP']:.3f}, CV_SHAP={med['CV_NND_SHAP']:.3f}, Ent_In={med['Entropy_Input']:.3f}")

print('\nSEARCH PEAKS (V4 Hybrid-US Seed 42):')
df = pd.read_csv(os.path.join(res_dir, 'tmp_v4_42', 'al_database_loop_49.csv')).iloc[30:]
print("Mean values of points explored by Hybrid-US:")
print(df.iloc[:, :5].mean())

print('\nSEARCH PEAKS (SUR Space-US Seed 42):')
df2 = pd.read_csv(os.path.join(res_dir, 'tmp_sur_42', 'al_database_loop_49.csv')).iloc[30:]
print("Mean values of points explored by Space-US:")
print(df2.iloc[:, :5].mean())
