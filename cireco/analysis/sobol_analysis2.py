import argparse
import numpy as np
import pandas as pd
from SALib.analyze import sobol
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--target", choices=["price", "symbiose"], default="price")
args = parser.parse_args()

problem = {
    'num_vars': 5,
    'names': ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost'],
    'bounds': [
        [0, 200],
        [0.1, 2],
        [1e-5, 1e-2],
        [0, 1],
        [0.1, 1]
    ]
}

csv1 = "sampling_lhs_raw.csv"
csv2 = "sampling_lhs_raw1.csv"

input_names = ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']
target_name = 'price'

# read and concatenate
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df = pd.concat([df1, df2], ignore_index=True)
# =============================================================================
csv3 = "sampling_saltelli_raw.csv"
df = pd.read_csv(csv3)
# 
# =============================================================================

# Input columns
X_cols = ['price_to_dispose','scarcity','density','cluster_spread','km_cost']

X = df[X_cols].values
y = df['price'].values

# Scale inputs for better kernel behavior
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kernel ridge regression with RBF kernel (Gaussian smoothing)
model = KernelRidge(alpha=1e-2, kernel='rbf', gamma=1.0)  # gamma controls bandwidth
model.fit(X_scaled, y)

# Smoothed predictions
y_smooth = model.predict(X_scaled)

# Replace original outputs with smoothed for Sobol analysis
df['price'] = y_smooth



df['density'] = np.log10(df['density'])

import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)



print(f"Total samples: {len(df)}")

# -------------------------
# Input distributions for OpenTURNS
# -------------------------
bounds = {
    'price_to_dispose': [0, 200],
    'scarcity': [0.1, 2],
    'density': [-5, -1],
    'cluster_spread': [0, 1],
    'km_cost': [0.1, 1]
}

df_results = df
df_mean = df_results.groupby(['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']).mean().reset_index()

import openturns as ot
import pandas as pd

# Assuming df_results is your DataFrame
input_names = ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']
target_name = 'symbiose'

# Convert inputs and outputs to OpenTURNS Sample
X_sample = ot.Sample(df_results[input_names].values.tolist())
Y_sample = ot.Sample(df_results[[target_name]].values.tolist())

# Now you can proceed with Sobol analysis
sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(X_sample, Y_sample, 1024)

S1=sensitivityAnalysis.getFirstOrderIndices()

ST=sensitivityAnalysis.getTotalOrderIndices()

# Convert OpenTURNS Point to a Python list
S1_list = list(S1)  # <-- this is the key step
ST_list = list(ST)  # <-- this is the key step

# Parameter names
input_names = ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']

# Optional: create a DataFrame
import pandas as pd
df_sobol = pd.DataFrame({
    'Parameter': input_names,
    'S1': S1_list,
    'ST': ST_list
})


# X-axis positions
x = np.arange(len(input_names))
width = 0.35  # bar width

# Create plot
plt.figure(figsize=(10,6))
plt.bar(x - width/2, S1_list, width, label='S1 (First-order)', color='skyblue')
#plt.bar(x + width/2, ST_list, width, label='ST (Total-order)', color='salmon')

# Labels
plt.ylabel('Sobol Indices')
plt.title("Sobol' Sensitivity Indices")
plt.xticks(x, input_names, rotation=45)
plt.ylim(np.min(S1), np.max( S1))
plt.legend()
plt.tight_layout()
plt.show()