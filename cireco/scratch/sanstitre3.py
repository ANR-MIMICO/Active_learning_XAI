# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 11:27:32 2025

@author: psaves
"""

# pce_sobol_with_salib.py
import numpy as np
import pandas as pd
import openturns as ot
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt

# --------------------------
# USER: data + problem spec
# --------------------------
# df_results must already be loaded (pandas DataFrame)
# Example (adapt to your variables and df):
# df_results = pd.read_csv("sobol_batch_raw_results.csv")
input_names = ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']
target_name = 'symbiose'

problem = {
    'num_vars': len(input_names),
    'names': input_names,
    'bounds': [
        [0, 200],
        [0.1, 2],
        [1e-5, 1e-2],
        [0, 1],
        [0.1, 1]
    ]
}

df_results = pd.read_csv("sampling_lhs_raw.csv")
df_results2 = pd.read_csv("sampling_lhs_raw1.csv")
df_results = pd.concat([df_results, df_results2], ignore_index=True)

df_results = pd.concat([df_results, df_results2], ignore_index=True)


df_results = df_results[df_results.iloc[:, 2] > 1e-3]

# --------------------------
# 1) Build PCE (Functional Chaos) from existing data
# --------------------------
# Prepare OT samples
X_np = df_results[input_names].values  # shape (n_samples, d)
Y_np = df_results[[target_name]].values  # shape (n_samples, 1)

X_ot = ot.Sample(X_np.tolist())
Y_ot = ot.Sample(Y_np.tolist())

# Create and run the PCE algorithm
# (OpenTURNS will automatically pick a basis/strategy if not specified)
algo = ot.FunctionalChaosAlgorithm(X_ot, Y_ot)
algo.run()
chaosResult = algo.getResult()

# Optional checks: quality/coeffs
#print("PCE: basis size =", chaosResult.getBasisSize())
if hasattr(chaosResult, "getMetaModel"):
    metamodel = chaosResult.getMetaModel()  # cheap surrogate (OpenTURNS Function)
else:
    raise RuntimeError("PCE did not return a metamodel; check chaosResult")

# --------------------------
# 2) Create Saltelli sample via SALib
# --------------------------
# Choose SALib base sample N (tradeoff: accuracy vs. cost).
# Note: the total rows fed to the metamodel will be len(param_values)
# SALib will create the required Saltelli design; check shape to confirm.
N = 4000  # start with 1024; increase for more accuracy (2k, 4k, etc.)
param_values = saltelli.sample(problem, N, calc_second_order=True)  # note: can be large
print("Generated Saltelli sample shape:", param_values.shape)

# --------------------------
# 3) Evaluate OT metamodel on Saltelli sample
# --------------------------
# Convert to OpenTURNS Sample and evaluate
param_ot = ot.Sample(param_values.tolist())
Y_pred_ot = metamodel(param_ot)              # returns an ot.Sample of shape (m,1)
Y_pred = np.array(Y_pred_ot)[:, 0]           # flatten to 1D numpy array

# Quick sanity checks
assert Y_pred.shape[0] == param_values.shape[0], "Mismatch between X and Y lengths"

# --------------------------
# 4) Analyze with SALib
# --------------------------
Si = sobol.analyze(problem, Y_pred, calc_second_order=True, print_to_console=True)

# Convert to DataFrame for neat display
df_sobol = pd.DataFrame({
    'Parameter': input_names,
    'S1': Si['S1'],
    'S1_conf': Si.get('S1_conf', [np.nan]*len(input_names)),
    'ST': Si['ST'],
    'ST_conf': Si.get('ST_conf', [np.nan]*len(input_names))
})

# Clip very small negatives to 0 for plotting (optional)
df_sobol[['S1','ST']] = df_sobol[['S1','ST']].clip(lower=0.0)

print(df_sobol)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DataFrame df_sobol déjà créé
# input_names = df_sobol['Parameter']

x = np.arange(len(input_names))
width = 0.35

# Bar plot S1 vs ST
fig, ax = plt.subplots(figsize=(10,6))
bars1 = ax.bar(x - width/2, df_sobol['S1'], width, label='S1 (First-order)', color='skyblue')
bars2 = ax.bar(x + width/2, df_sobol['ST'], width, label='ST (Total-order)', color='salmon')

# Ajouter labels sur chaque barre
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0,3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0,3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# Axes, titre, légende
ax.set_ylabel("Sobol index")
ax.set_ylim(0, 1)
ax.set_xticks(x)
ax.set_xticklabels(input_names, rotation=45, ha='right')
ax.set_title("Sobol Sensitivity Indices (S1 vs ST)")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# --------------------------
# Optionnel: Heatmap S2 (interactions du second ordre)
# --------------------------
import seaborn as sns

S2 = Si['S2']  # matrice (d x d)
S2_conf = Si.get('S2_conf', None)

# Clip petits négatifs
S2 = np.clip(S2, 0, 1)

plt.figure(figsize=(8,6))
sns.heatmap(S2, annot=True, xticklabels=input_names, yticklabels=input_names,
            cmap="Reds", cbar_kws={'label': 'S2 (second-order indices)'}, fmt=".2f")
plt.title("Second-order Sobol indices heatmap")
plt.tight_layout()
plt.show()
