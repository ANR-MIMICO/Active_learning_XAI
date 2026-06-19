# -*- coding: utf-8 -*-
"""
Full PCE + Sobol + PDP/ICE analysis for 5 variables
Includes: price_to_dispose, scarcity, density, cluster_spread, km_cost
"""

import numpy as np
import pandas as pd
import openturns as ot
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.preprocessing import MinMaxScaler

from smt.design_space import (
    DesignSpace,
    FloatVariable,
    CategoricalVariable,
    )
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.cm as cm
from sklearn.inspection import partial_dependence
import matplotlib.colors as mcolors
import warnings
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")
from smt.sampling_methods import LHS
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# -------------------------
# User: data files and names
# -------------------------
csv1 = "Newbounds.csv"
input_names = ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']
input_names_all = ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']

target_name = 'price' #symbiose


# read and concatenate
df = pd.read_csv(csv1)
df = df.groupby(input_names).mean().reset_index()

df['density'] = np.log10(df['density'])
df['scarcity'] = np.log2(df['scarcity'])
df_orig = df.copy(True)
# -------------------------
# Input distributions for OpenTURNS
# -------------------------
bounds = {
    'price_to_dispose': [0, 200],
    'scarcity': [-2, 2],
    'density': [-5, -1],
    'cluster_spread': [0, 0.5],
    'km_cost': [0, 10]
}
ds = DesignSpace(
    [
        FloatVariable(0, 200),
        FloatVariable(-2, 2),
        FloatVariable(-5, -1),
        FloatVariable(0, 0.5),
        FloatVariable(0, 10),

    ]
)
# -------------------------
# Define bins
# -------------------------
bin_edges = [0.0, -4, -2, np.inf]
# label names
bin_names = ['very_low', 'mid', 'high']

# create boolean subsets
df_very_low = df[df['density'] < -4].reset_index(drop=True)
df_mid      = df[(df['density'] >= -4) & (df['density'] < -2)].reset_index(drop=True)
df_high     = df[df['density'] >= -2].reset_index(drop=True)



import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_orig = pd.DataFrame(scaler.fit_transform(df_orig), columns=df_orig.columns)
df_very_low = pd.DataFrame(scaler.fit_transform(df_very_low), columns=df_very_low.columns)
df_mid = pd.DataFrame(scaler.fit_transform(df_mid), columns=df_mid.columns)
df_high = pd.DataFrame(scaler.fit_transform(df_high), columns=df_high.columns)

print(f"Total samples: {len(df)}  |  very_low (<1e-4): {len(df_very_low)}  |  mid (1e-4-1e-2): {len(df_mid)}  |  high (>=1e-2): {len(df_high)}")

# Warnings if too small
min_samples_warn = 30
for name, dsub in zip(bin_names, [df_very_low, df_mid, df_high]):
    if len(dsub) < min_samples_warn:
        print(f"WARNING: subset '{name}' has only {len(dsub)} samples. PCE / Sobol may be unstable.")

# -------------------------
# Retirer la variable density de l'analyse
# -------------------------
input_names = [n for n in input_names_all if n != 'density']

d = len(input_names)
print(f"Variables utilisées ({d}): {input_names}")



problem = {
    'num_vars': d,
    'names': input_names,
    'bounds': [[0,1] for n in input_names]
}

# Construire la distribution uniforme multivariée pour les inputs (hors density)
marginals = [ot.Uniform(0,1) for name in input_names]
input_dist = ot.ComposedDistribution(marginals)


    
def build_pce_metamodel(X_np,Y_np, input_names, target_name, input_dist):
    if X_np.shape[0] < 5:
        raise RuntimeError("Trop peu d'échantillons pour construire un PCE.")

    X_ot = ot.Sample(X_np)
    Y_ot = ot.Sample(np.atleast_2d(Y_np).T)

    degree = 25
    enum = ot.HyperbolicAnisotropicEnumerateFunction(len(input_names), 0.85)
    poly_coll = ot.OrthogonalProductPolynomialFactory(
        [ot.LegendreFactory()] * len(input_names),
        enum
    )
    trunc = ot.FixedStrategy(poly_coll, degree)
    proj = ot.LeastSquaresStrategy(X_ot, Y_ot)

    algo = ot.FunctionalChaosAlgorithm(X_ot, Y_ot, input_dist, trunc, proj)
    try:
        algo.setMaximumEvaluationNumber(int(1e6))
    except Exception:
        pass
    algo.run()
    chaosResult = algo.getResult()
    if not hasattr(chaosResult, "getMetaModel"):
        raise RuntimeError("PCE n'a pas retourné de metamodel ; vérifier les données / OpenTURNS.")
    return chaosResult.getMetaModel()
class PCEWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, ot_metamodel):
        self.model = ot_metamodel

    def fit(self, X=None, y=None):
        self.n_features_in_ = X.shape[1] if X is not None else None
        self.fitted_ = True
        return self

    def predict(self, X):
        if not hasattr(self, "fitted_"):
            raise ValueError("This PCEWrapper instance is not fitted yet. Call fit() first.")
        X_ot = ot.Sample(X.values)
        Y_ot = self.model(X_ot)
        return np.array(Y_ot)[:,0]




def build_mlp_metamodel(df_sub,X_test, input_names, target_name, random_state=0):
    """
    Build a scikit-learn MLP metamodel from a DataFrame subset.

    Parameters
    ----------
    df_sub : pd.DataFrame
        Subset of the data containing input features and target.
    input_names : list of str
        Names of input features.
    target_name : str
        Name of the target column.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    mlp_model : sklearn Pipeline
        Trained MLP model with scaler.
    r2_test : float
        R² score on test set.
    rmse_test : float
        RMSE on test set.
    """
    # Prepare data
    X = df_sub[input_names].values
    y = df_sub[target_name].values

    # Define pipeline with scaling + MLP
    mlp_model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 50,50,32),
            activation='relu',
            solver='adam',
            max_iter=2222,
            random_state=random_state
        ))
    ])



    # Train
    mlp_model.fit(X, y)

    # Evaluate
    y_pred = mlp_model.predict(X_test)
    return y_pred


from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# -------------------------
# Shuffle and split
# -------------------------

x2=  df_high[input_names].values
#   x[:,2] = np.log10(x[:,2])

y2=  df_high[target_name].values


#x2 = df_orig[input_names_all].values
#input_names = input_names_all

#y2 = df_orig[target_name].values

from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()
x2 = scaler.fit_transform(x2)

X_shuf, y_shuf = shuffle(x2, y2, random_state=0)

X_train = X_shuf[:4000]
y_train = y_shuf[:4000]
X_test  = X_shuf[-1000:]
y_test  = y_shuf[-1000:]

# -------------------------
# Build PCE metamodel on training set
# -------------------------
df_train = pd.DataFrame(X_train, columns=input_names)
df_train[target_name] = y_train

pce_model = build_pce_metamodel(df_train, input_names, target_name)

# -------------------------
# Evaluate on test set
# -------------------------
import openturns as ot

X_test_ot = ot.Sample(X_test.tolist())
    
try : 
    y_pred = np.array(pce_model(X_test_ot))[:,0]
    print("pce")
    namemeta = "pce"
except :
    print("mlp")
    namemeta = "mlp"
    y_pred = build_mlp_metamodel(df_train,X_test, input_names_all, target_name)
    
# -------------------------
# Metrics
# -------------------------
r2 = r2_score(y_test, y_pred)

print(f"PCE Test R²  : {r2:.4f}")

# Optional: plot predicted vs true
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True")
plt.ylabel("Prediction")
plt.title(namemeta+"Metamodel Validation")
plt.grid(True)
plt.show()




# -------------------------
# Build PCE for each bin
# -------------------------
metamodel_vlow = build_pce_metamodel(df_very_low, input_names, target_name) if len(df_very_low)>0 else None
metamodel_mid  = build_pce_metamodel(df_mid, input_names, target_name)      if len(df_mid)>0 else None
metamodel_high = build_pce_metamodel(df_high, input_names, target_name)     if len(df_high)>0 else None
print("PCE built for available bins (None means not enough samples).")

# -------------------------
# Saltelli sample (d variables)
# -------------------------
N = 1024  # base sample size (ajuste selon précision désirée)
param_values = saltelli.sample(problem, N, calc_second_order=True)
print("Saltelli sample shape:", param_values.shape)

# Evaluate metamodels (skip those that are None)
param_ot = ot.Sample(param_values.tolist())

def eval_metamodel_or_nan(metamodel, param_ot):
    if metamodel is None:
        return np.full(len(param_ot), np.nan)
    return np.array(metamodel(param_ot))[:, 0]

Y_vlow = eval_metamodel_or_nan(metamodel_vlow, param_ot)
Y_mid  = eval_metamodel_or_nan(metamodel_mid, param_ot)
Y_high = eval_metamodel_or_nan(metamodel_high, param_ot)

# -------------------------
# Analyse Sobol (handle nan by skipping)
# -------------------------
def safe_sobol(Y_pred):
    if np.isnan(Y_pred).all():
        return None
    # if some NaNs present, raise or mask — here we require full vector
    if np.isnan(Y_pred).any():
        raise RuntimeError("Metamodel returned NaNs; cannot run Sobol analyze.")
    return sobol.analyze(problem, Y_pred, calc_second_order=True, print_to_console=False)

Si_vlow = safe_sobol(Y_vlow)
Si_mid  = safe_sobol(Y_mid)
Si_high = safe_sobol(Y_high)



# Convert to DataFrames when present
def sobol_to_df(Si, names):
    if Si is None:
        # return dataframe of NaNs
        return pd.DataFrame({
            'Parameter': names,
            'S1': [np.nan]*len(names),
            'S1_conf': [np.nan]*len(names),
            'ST': [np.nan]*len(names),
            'ST_conf': [np.nan]*len(names)
        })
    df_s = pd.DataFrame({
        'Parameter': names,
        'S1': Si['S1'],
        'S1_conf': Si.get('S1_conf', [np.nan]*len(names)),
        'ST': Si['ST'],
        'ST_conf': Si.get('ST_conf', [np.nan]*len(names))
    })
    df_s[['S1','ST']] = df_s[['S1','ST']].clip(lower=0.0)
    return df_s

df_vlow_sobol = sobol_to_df(Si_vlow, input_names)
df_mid_sobol  = sobol_to_df(Si_mid,  input_names)
df_high_sobol = sobol_to_df(Si_high, input_names)

print("\nSobol (very_low):\n", df_vlow_sobol)
print("\nSobol (mid):\n", df_mid_sobol)
print("\nSobol (high):\n", df_high_sobol)

# -------------------------
# Plot : 6 barres par variable (S1_vlow, ST_vlow, S1_mid, ST_mid, S1_high, ST_high)
# -------------------------
x = np.arange(d)
width = 0.11  # small width since 6 bars

# values arrays (convert to numeric arrays, replace nan with 0 for plotting but annotate NaN)
s1_vlow = df_vlow_sobol['S1'].values
st_vlow = df_vlow_sobol['ST'].values
s1_mid  = df_mid_sobol['S1'].values
st_mid  = df_mid_sobol['ST'].values
s1_high = df_high_sobol['S1'].values
st_high = df_high_sobol['ST'].values

# colors chosen: blue (very low), orange (mid), green (high) — light/dark pairs
# colors: light/dark magenta, light/dark blue, light/dark cyan
colors = {
    'S1_vlow':  "#F4A8FF",  # light magenta
    'ST_vlow':  "#8B008B",  # dark magenta
    'S1_mid' :  "#89CFF0",  # light blue
    'ST_mid' :  "#1F4E79",  # dark blue
    'S1_high':  "#7FFFD4",  # light cyan (aquamarine)
    'ST_high':  "#008B8B"   # dark cyan
}

fig, ax = plt.subplots(figsize=(14,6))

# positions: -2.5w, -1.5w, -0.5w, +0.5w, +1.5w, +2.5w
ax.bar(x - 2.5*width, s1_vlow, width, label='S1 (very_low)', color=colors['S1_vlow'])
ax.bar(x - 1.5*width, st_vlow, width, label='ST (very_low)', color=colors['ST_vlow'])
ax.bar(x - 0.5*width, s1_mid,  width, label='S1 (mid)',      color=colors['S1_mid'])
ax.bar(x + 0.5*width, st_mid,  width, label='ST (mid)',      color=colors['ST_mid'])
ax.bar(x + 1.5*width, s1_high, width, label='S1 (high)',     color=colors['S1_high'])
ax.bar(x + 2.5*width, st_high, width, label='ST (high)',     color=colors['ST_high'])

# annotations (show value or 'n/a' if NaN)
def annotate(values, positions):
    for val, xpos in zip(values, positions):
        if np.isnan(val):
            txt = "n/a"
            y = 0.02
            ax.annotate(txt, xy=(xpos, y), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=8, color='red')
        else:
            ax.annotate(f"{val:.2f}", xy=(xpos, val), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

annotate(s1_vlow, x - 2.5*width)
annotate(st_vlow, x - 1.5*width)
annotate(s1_mid,  x - 0.5*width)
annotate(st_mid,  x + 0.5*width)
annotate(s1_high, x + 1.5*width)
annotate(st_high, x + 2.5*width)

# axes & legend
ax.set_xticks(x)
ax.set_xticklabels(input_names, rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.set_ylabel("Sobol index")
ax.set_title("Sobol indices (S1 & ST) across three density for price")
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# =============================================================================
# # Save outputs
# fig.savefig("sobol_6bars_three_bins.png", dpi=300)
# df_vlow_sobol.to_csv("sobol_very_low.csv", index=False)
# df_mid_sobol.to_csv("sobol_mid.csv", index=False)
# df_high_sobol.to_csv("sobol_high.csv", index=False)
# =============================================================================
