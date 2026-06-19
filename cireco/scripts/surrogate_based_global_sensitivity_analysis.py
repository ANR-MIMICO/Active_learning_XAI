import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
)
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import partial_dependence
import warnings
import sys

warnings.filterwarnings("ignore")
from smt.sampling_methods import LHS
import itertools as it
from scipy.ndimage import gaussian_filter1d

# -------------------------
# User: data files and names
# -------------------------
csv1 = sys.argv[1]
input_names_all = [
    "price_to_dispose",
    "scarcity",
    "density",
    "cluster_spread",
    "km_cost",
]
input_names = ["price_to_dispose", "scarcity", "cluster_spread", "km_cost"]

target_name = "symbiose"  # symbiose, price


# read and concatenate
df = pd.read_csv(csv1)
df = df.groupby(input_names).mean().reset_index()

df["density"] = np.log10(df["density"])
df["scarcity"] = np.log2(df["scarcity"])
df_orig = df.copy(True)

# -------------------------
# Define bins
# -------------------------
bin_edges = [-3, -2]
# label names
bin_names = ["very_low", "mid", "high"]

# create boolean subsets
df_very_low = df[df["density"] < bin_edges[0]].reset_index(drop=True)
df_mid = df[
    (df["density"] >= bin_edges[0]) & (df["density"] < bin_edges[1])
].reset_index(drop=True)
df_high = df[df["density"] >= bin_edges[1]].reset_index(drop=True)

# seuil = 3
# df = df[df['density'] > -seuil].reset_index(drop=True)
# df = df[df['density'] < -seuil+0.5].reset_index(drop=True)

print(f"Total samples: {len(df)}")

# -------------------------
# Input distributions for OpenTURNS
# -------------------------
bounds = {
    "price_to_dispose": [0, 200],
    "scarcity": [-2, 2],
    "density": [-4, -1],
    "cluster_spread": [0, 0.5],
    "km_cost": [0, 2],
}
ds = DesignSpace(
    [
        FloatVariable(0, 200),
        FloatVariable(-2, 2),
        FloatVariable(-4, -1),
        FloatVariable(0, 0.5),
        FloatVariable(0, 2),
    ]
)


def build_pce_metamodel(X_np, Y_np, input_names, target_name, input_dist):
    if X_np.shape[0] < 5:
        raise RuntimeError("Trop peu d'échantillons pour construire un PCE.")

    X_ot = ot.Sample(X_np)
    Y_ot = ot.Sample(np.atleast_2d(Y_np).T)

    degree = 150
    enum = ot.HyperbolicAnisotropicEnumerateFunction(len(input_names), 0.999999)
    poly_coll = ot.OrthogonalProductPolynomialFactory(
        [ot.LegendreFactory()] * len(input_names), enum
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
        raise RuntimeError(
            "PCE n'a pas retourné de metamodel ; vérifier les données / OpenTURNS."
        )
    return chaosResult.getMetaModel(), chaosResult


class PCEWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, ot_metamodel):
        self.model = ot_metamodel

    def fit(self, X=None, y=None):
        self.n_features_in_ = X.shape[1] if X is not None else None
        self.fitted_ = True
        return self

    def predict(self, X):
        if not hasattr(self, "fitted_"):
            raise ValueError(
                "This PCEWrapper instance is not fitted yet. Call fit() first."
            )
        X_ot = ot.Sample(X.values)
        Y_ot = self.model(X_ot)
        return np.array(Y_ot)[:, 0]


def all_sobol_indices(pce_result, input_names):
    """
    Compute all Sobol' indices from a PCE metamodel.

    Parameters
    ----------
    pce_result : ot.FunctionalChaosResult
        The fitted PCE result from OpenTURNS.
    input_names : list of str
        Names of the input variables.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with columns [order, variables, Sobol_index].
    """
    sensitivity = ot.FunctionalChaosSobolIndices(pce_result)
    d = len(input_names)
    records = []
    # Loop over all subset sizes (1 to d)
    for order in range(1, d + 1):
        for idx in it.combinations(range(d), order):
            # print(idx)
            if order == 1:
                val = sensitivity.getSobolIndex(idx[0])
            elif order == 2:
                val = sensitivity.getSobolIndex([idx[0], idx[1]])
            else:
                # must pass as ot.Indices
                val = sensitivity.getSobolIndex(ot.Indices(idx))
            vars_ = [input_names[i] for i in idx]
            records.append({"order": order, "variables": vars_, "Sobol_index": val})

    # First-order and total-order separately (more efficient API calls)
    for i in range(d):
        ST = sensitivity.getSobolTotalIndex(i)
        records.append(
            {"order": "Total", "variables": [input_names[i]], "Sobol_index": ST}
        )

    return pd.DataFrame(records)


x = df[input_names_all].values
y = df[target_name].values
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

marginals = [ot.Uniform(0, 1) for name in input_names_all]
input_dist = ot.ComposedDistribution(marginals)
metamodel, _ = build_pce_metamodel(x, y, input_names_all, target_name, input_dist)
print("PCE metamodel constructed.")
pce_wrapper = PCEWrapper(metamodel)
pce_wrapper.fit(x, y)
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 50, 50, 32),
    activation="relu",
    solver="adam",
    max_iter=2222,
    random_state=0,
)
mlp.fit(x, y)


# Create LHS sampler
lhs = LHS(xlimits=np.array(ds.get_x_limits()), criterion="ese", random_state=42)
# Generate 100 samples
n_samples = 250
X_lhs = lhs(n_samples)
scaler = MinMaxScaler()
X_lhs = scaler.fit_transform(X_lhs)
X_lhs = pd.DataFrame(X_lhs, columns=input_names_all)


# -------------------------
# Build PCE for each bin
# -------------------------
problem = {
    "num_vars": len(input_names_all),
    "names": input_names_all,
    "bounds": [[0, 1] for n in input_names_all],
}


xvl = df_very_low[input_names_all].values
yvl = df_very_low[target_name].values
scaler = MinMaxScaler()
xvl = scaler.fit_transform(xvl)
metamodel_vlow, chaos_vlow = (
    build_pce_metamodel(xvl, yvl, input_names_all, target_name, input_dist)
    if len(df_very_low) > 0
    else None
)
metamodel_vlow = MLPRegressor(
    hidden_layer_sizes=(128, 50, 50, 32),
    activation="relu",
    solver="adam",
    max_iter=2333,
    random_state=0,
)
metamodel_vlow.fit(xvl, yvl)


xm = df_mid[input_names_all].values
ym = df_mid[target_name].values
scaler = MinMaxScaler()
xm = scaler.fit_transform(xm)
metamodel_mid, chaos_mid = (
    build_pce_metamodel(xm, ym, input_names_all, target_name, input_dist)
    if len(df_very_low) > 0
    else None
)
metamodel_mid = MLPRegressor(
    hidden_layer_sizes=(128, 50, 50, 32),
    activation="relu",
    solver="adam",
    max_iter=2222,
    random_state=0,
)
metamodel_mid.fit(xm, ym)


xh = df_high[input_names_all].values
yh = df_high[target_name].values
scaler = MinMaxScaler()
xh = scaler.fit_transform(xh)
metamodel_high, chaos_high = (
    build_pce_metamodel(xh, yh, input_names_all, target_name, input_dist)
    if len(df_very_low) > 0
    else None
)
metamodel_high = MLPRegressor(
    hidden_layer_sizes=(128, 50, 50, 32),
    activation="relu",
    solver="adam",
    max_iter=2222,
    random_state=0,
)
metamodel_high.fit(xh, yh)


# Example usage


metamodel_all, chaos_all = build_pce_metamodel(
    x, y, input_names_all, target_name, input_dist
)

df_sobol = all_sobol_indices(chaos_all, input_names_all)
df_sobol_sorted = df_sobol[df_sobol["order"] != "Total"]
df_sobol_sorted = df_sobol_sorted.sort_values(
    by="Sobol_index", ascending=False
).reset_index(drop=True)
print("\nSobol (all data):\n", df_sobol_sorted)


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
    try:
        return np.array(metamodel(param_ot))[:, 0]
    except:
        return np.array(metamodel.predict(np.array(param_ot)))


Y_vlow = eval_metamodel_or_nan(metamodel_vlow, param_ot)
Y_mid = eval_metamodel_or_nan(metamodel_mid, param_ot)
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
    return sobol.analyze(
        problem, Y_pred, calc_second_order=True, print_to_console=False
    )


Si_vlow = safe_sobol(Y_vlow)
Si_mid = safe_sobol(Y_mid)
Si_high = safe_sobol(Y_high)


# Convert to DataFrames when present
def sobol_to_df(Si, names):
    if Si is None:
        # return dataframe of NaNs
        return pd.DataFrame(
            {
                "Parameter": names,
                "S1": [np.nan] * len(names),
                "S1_conf": [np.nan] * len(names),
                "ST": [np.nan] * len(names),
                "ST_conf": [np.nan] * len(names),
            }
        )
    df_s = pd.DataFrame(
        {
            "Parameter": names,
            "S1": Si["S1"],
            "S1_conf": Si.get("S1_conf", [np.nan] * len(names)),
            "ST": Si["ST"],
            "ST_conf": Si.get("ST_conf", [np.nan] * len(names)),
        }
    )
    df_s[["S1", "ST"]] = df_s[["S1", "ST"]].clip(lower=0.0)
    return df_s


df_vlow_sobol = sobol_to_df(Si_vlow, input_names_all)
df_mid_sobol = sobol_to_df(Si_mid, input_names_all)
df_high_sobol = sobol_to_df(Si_high, input_names_all)

print("\nSobol (very_low):\n", df_vlow_sobol)
print("\nSobol (mid):\n", df_mid_sobol)
print("\nSobol (high):\n", df_high_sobol)

# -------------------------
# Plot : 6 barres par variable (S1_vlow, ST_vlow, S1_mid, ST_mid, S1_high, ST_high)
# -------------------------
xplot = np.arange(len(input_names))
width = 0.11  # small width since 6 bars

# values arrays (convert to numeric arrays, replace nan with 0 for plotting but annotate NaN)
s1_vlow = df_vlow_sobol["S1"].values
st_vlow = df_vlow_sobol["ST"].values
s1_mid = df_mid_sobol["S1"].values
st_mid = df_mid_sobol["ST"].values
s1_high = df_high_sobol["S1"].values
st_high = df_high_sobol["ST"].values

# colors chosen: blue (very low), orange (mid), green (high) — light/dark pairs
# colors: light/dark magenta, light/dark blue, light/dark cyan
colors = {
    "S1_vlow": "#F4A8FF",  # light magenta
    "ST_vlow": "#8B008B",  # dark magenta
    "S1_mid": "#89CFF0",  # light blue
    "ST_mid": "#1F4E79",  # dark blue
    "S1_high": "#7FFFD4",  # light cyan (aquamarine)
    "ST_high": "#008B8B",  # dark cyan
}

fig, ax = plt.subplots(figsize=(14, 6))


def drop_variable(values, names, varname="density"):
    """Retire varname d'un tableau numpy et de la liste de noms correspondante"""
    mask = np.array(names) != varname
    return values[mask], [n for n in names if n != varname]


s1_vlow, input_names_no_density = drop_variable(s1_vlow, input_names_all, "density")
st_vlow, _ = drop_variable(st_vlow, input_names_all, "density")
s1_mid, _ = drop_variable(s1_mid, input_names_all, "density")
st_mid, _ = drop_variable(st_mid, input_names_all, "density")
s1_high, _ = drop_variable(s1_high, input_names_all, "density")
st_high, _ = drop_variable(st_high, input_names_all, "density")

# positions: -2.5w, -1.5w, -0.5w, +0.5w, +1.5w, +2.5w
ax.bar(
    xplot - 2.5 * width, s1_vlow, width, label="S1 (very_low)", color=colors["S1_vlow"]
)
ax.bar(
    xplot - 1.5 * width, st_vlow, width, label="ST (very_low)", color=colors["ST_vlow"]
)
ax.bar(xplot - 0.5 * width, s1_mid, width, label="S1 (mid)", color=colors["S1_mid"])
ax.bar(xplot + 0.5 * width, st_mid, width, label="ST (mid)", color=colors["ST_mid"])
ax.bar(xplot + 1.5 * width, s1_high, width, label="S1 (high)", color=colors["S1_high"])
ax.bar(xplot + 2.5 * width, st_high, width, label="ST (high)", color=colors["ST_high"])


# annotations (show value or 'n/a' if NaN)
def annotate(values, positions):
    for val, xpos in zip(values, positions):
        if np.isnan(val):
            txt = "n/a"
            y = 0.02
            ax.annotate(
                txt,
                xy=(xpos, y),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="red",
            )
        else:
            ax.annotate(
                f"{val:.2f}",
                xy=(xpos, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


annotate(s1_vlow, xplot - 2.5 * width)
annotate(st_vlow, xplot - 1.5 * width)
annotate(s1_mid, xplot - 0.5 * width)
annotate(st_mid, xplot + 0.5 * width)
annotate(s1_high, xplot + 1.5 * width)
annotate(st_high, xplot + 2.5 * width)

# axes & legend
ax.set_xticks(xplot)
ax.set_xticklabels(input_names, rotation=45, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Sobol index")
ax.set_title("Sobol indices (S1 & ST) across three density for" + target_name)
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()


# We need one subplot per feature except 'density'
features_to_plot = [f for f in input_names]
n_features = len(features_to_plot)

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()  # flatten for easy indexing

density_values = X_lhs["density"].values
norm = plt.Normalize(density_values.min(), density_values.max())
cmap = plt.cm.jet  # blue → red

for i, feat in enumerate(features_to_plot):
    ax = axes[i]
    pd_results = partial_dependence(
        mlp, X_lhs, [feat], kind="both", grid_resolution=100
    )
    grid = pd_results["grid_values"][0]
    pdp = pd_results["average"][0]
    ice = pd_results["individual"][0]  # shape (n_samples, grid_resolution)

    sigma = 3  # adjust smoothing
    ice = gaussian_filter1d(ice, sigma=sigma, axis=1)  # smooth along grid axis

    # Plot ICE curves colored by density
    for j in range(ice.shape[0]):
        color = cmap(norm(density_values[j]))
        ax.plot(grid, ice[j, :], color=color, alpha=0.5)

    # Plot PDP curve thicker
    ax.plot(grid, pdp, color="black", linewidth=5)
    ax.set_title(target_name + f" PDP/ICE {feat}")

# Remove any empty subplots if n_features < 4
for k in range(n_features, 4):
    fig.delaxes(axes[k])

# Place a single vertical colorbar on the right side
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="Density")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
plt.show()
