# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 11:35:17 2025

@author: psaves
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Example: your dataset
input_names = ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']
target_name = 'price'
df_results = pd.read_csv("sobol_batch_raw_results.csv")

X = df_results[input_names].values
y = df_results[target_name].values

# Train Random Forest surrogate
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X, y)

from SALib.sample import saltelli
from SALib.analyze import sobol

problem = {
    'num_vars': len(input_names),
    'names': input_names,
    'bounds': [
        [0, 200],       # price_to_dispose
        [0.1, 2],       # scarcity
        [0.001, 0.1],   # density
        [0, 1],         # cluster_spread
        [0.1, 1]        # km_cost
    ]
}

N = 1024  # base sample size
param_values = saltelli.sample(problem, N, calc_second_order=True)

# Predict with the trained Random Forest
Y_pred = rf.predict(param_values)


Si = sobol.analyze(problem, Y_pred, calc_second_order=True, print_to_console=True)

# Convert to DataFrame for plotting
import matplotlib.pyplot as plt
import pandas as pd

df_sobol = pd.DataFrame({
    'Parameter': input_names,
    'S1': Si['S1'],
    'ST': Si['ST']
})

# Clip small negative values for plotting
df_sobol[['S1','ST']] = df_sobol[['S1','ST']].clip(lower=0.0)
print(df_sobol)


import numpy as np
x = np.arange(len(input_names))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar(x - width/2, df_sobol['S1'], width, label='S1 (First-order)', color='skyblue')
plt.bar(x + width/2, df_sobol['ST'], width, label='ST (Total-order)', color='salmon')
plt.xticks(x, df_sobol['Parameter'], rotation=45, ha='right')
plt.ylabel('Sobol indices')
plt.ylim(0, 1)
plt.title('Sobol indices from Random Forest surrogate')
plt.legend()
plt.tight_layout()
plt.show()
