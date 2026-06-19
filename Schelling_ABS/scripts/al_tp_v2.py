import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:27:23 2024

@author: psaves
"""
from scipy.stats import gaussian_kde
import time
import asyncio
from gama_client.message_types import MessageTypes
from gama_client.sync_client import GamaSyncClient
from typing import Dict
import csv
from sklearn.neighbors import NearestNeighbors
import os
import os
import subprocess
import win32com.client
import signal
import time
import sys
import fileinput
import matplotlib.pyplot as plt
import numpy as np

from smt.applications import EGO
from smt.design_space import DesignSpace
from smt.surrogate_models import KRG

import shap

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor # Use MLPRegressor if predicting the float value
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

from smt.surrogate_models import KRG, KPLS
from smt.applications.mixed_integer import (
    MixedIntegerSamplingMethod,
)
from smt.sampling_methods import LHS, Random, FullFactorial
from smt.surrogate_models import MixIntKernelType

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
#from smt.utils import compute_rms_error
import re
import json
import shutil
import unittest

from smt.sampling_methods import LHS
from smt.surrogate_models import MixIntKernelType
from smt_design_space_ext import (
    AdsgDesignSpaceImpl,
    ConfigSpaceDesignSpaceImpl,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
    CategoricalVariable,
)
import os
from scipy.stats import norm
import sys

import matplotlib.pyplot as plt
import numpy as np

from smt.sampling_methods import LHS
from smt.applications.mfk import MFK, NestedLHS
import seaborn as sns


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import minimize

def _find_best_point(self, x_data=None, y_data=None, enable_tunneling=False):
    """
    Function that analyse a set of x_data and y_data and give back the
    more interesting point to evaluates according to the selected criterion

    Parameters
    ----------

    x_data: ndarray(n_points, nx)
    y_data: ndarray(n_points, 1)

    Returns
    -------

    ndarray(nx, 1): the next best point to evaluate
    boolean: success flag

    """



def explicit_ei(x, model, y_min, return_grad=False):
    """
    Calculates EI and its gradient (optional) for a given point x.
    """
    x = np.atleast_2d(x)
    
    # 1. Predict Mean and Variance
    # SMT returns (n, 1), we flatten to (n,)
    mu = model.predict_values(x).flatten()
    sigma2 = model.predict_variances(x).flatten()
    
    # Handle numerical stability for sigma
    sigma = np.sqrt(np.maximum(sigma2, 1e-9)) 
    
    # 2. Calculate Z score
    with np.errstate(divide='ignore'):
        z = (y_min - mu) / sigma
    
    # 3. Calculate EI
    ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Zero out EI where sigma is effectively zero (avoid divide by zero errors)
    ei[sigma < 1e-9] = 0.0
    
    
    return ei



def custom_beeswarm_plot(shap_vals, feature_vals, feature_names, title="SHAP Beeswarm Plot"):
    """
    Reproduces the SHAP summary beeswarm plot using standard Matplotlib.
    
    Args:
        shap_vals (np.array): Shape (N_samples, N_features). 
                              Ensure you pass the array for the specific class 
                              (e.g., shap_values[1] if it's a classifier).
        feature_vals (np.array): Shape (N_samples, N_features). The original input data.
        feature_names (list): List of feature names strings.
    """
    
    # 1. Compute Feature Importance (Mean Absolute SHAP Value)
    # We sort features so the most important one is at the top
    feature_importance = np.abs(shap_vals).mean(axis=0)
    sorted_indices = np.argsort(feature_importance) # Ascending order
    
    # Prepare the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Constants for visual style
    row_height = 1.0
    y_ticks = []
    y_labels = []
    
    # Colormap (Red=High feature value, Blue=Low feature value)
    cmap = plt.get_cmap('coolwarm') # or 'RdBu_r'
    
                    
    # Parameters
    n_samples, n_features = shap_vals.shape

    density_matrix = np.zeros((n_features, n_samples))
    for l in range(n_features):
        feature_col = shap_vals[:, l]
        kde = gaussian_kde(feature_col)
        densities = kde(feature_col)
        if densities.max() > 0:
            densities = densities / densities.max()
        density_matrix[l, :] = densities
    density_matrix = (1-density_matrix).T
    density_matrix = density_matrix/np.exp(np.abs(shap_vals))


    density_mean_matrix = np.array([np.mean(density_matrix,axis=1)]*n_features).T

        
    # 2. Loop through features (from bottom to top importance)
    for i, idx in enumerate(sorted_indices):
        
        # Get data for this feature
        s_vals = shap_vals[:, idx]
        f_vals = feature_vals[:, idx]
        f_vals= density_mean_matrix[:,idx]

        # Normalize feature values to 0-1 for coloring
        # Handle case where variance is 0 to avoid div by zero
        if np.max(f_vals) - np.min(f_vals) > 0:
            norm_f_vals = (f_vals - np.min(f_vals)) / (np.max(f_vals) - np.min(f_vals))
        else:
            norm_f_vals = np.zeros_like(f_vals) + 0.5 # Grey if constant

        # 3. Calculate Jitter (Vertical Spread) based on density
        # We use a histogram to estimate density at different SHAP values
        # and spread points vertically where density is high.
        nbins = 50
        hist, bin_edges = np.histogram(s_vals, bins=nbins)
        bin_indices = np.digitize(s_vals, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, nbins-1)
        
        # Calculate a random vertical offset, scaled by the density of that bin
        jitter = np.zeros_like(s_vals)
        for bin_i in range(nbins):
            count = hist[bin_i]
            if count > 1:
                # Find points in this bin
                mask = (bin_indices == bin_i)
                # Spread them out. Scale spread by count to mimic "swarm"
                # Max spread is restricted to avoid overlapping rows too much
                spread_width = min(0.4, 0.01 * count) 
                jitter[mask] = np.random.uniform(-spread_width, spread_width, size=count)
        
        # 4. Plot the points
        # Y-position = current_row_index + jitter
        y_pos = i + jitter
        
        # Scatter plot
        sc = ax.scatter(s_vals, y_pos, 
                        c=norm_f_vals, 
                        cmap=cmap, 
                        s=15,         # Dot size
                        alpha=0.8,    # Transparency
                        edgecolors='none')
        
        y_ticks.append(i)
        y_labels.append(feature_names[idx])

    # 5. Formatting
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add vertical line at 0
    ax.axvline(x=0, color="#999999", linestyle="-", linewidth=1, zorder=-1)
    
    # Clean up spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0) # Hide y ticks markers
    
    # 6. Add Colorbar
    # Create a dummy mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, aspect=40)
    cbar.set_label('Feature value', size=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    
    plt.tight_layout()
    plt.show()

# --- USAGE ---


current_file_path = os.path.abspath(__file__)[-15]
# Specify the directory containing the files
folder_path = current_file_path[:-15]+'doe_5_200'

DATABASEy = []
DATABASEx = []
# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Make sure it's a file and not a directory
    if os.path.isfile(file_path):
        # Open the file
        with open(file_path, 'r') as file:
            last_line = file.readlines()[-1]
            DATABASEx.append(np.array(file_path[10:-11].split('_'),dtype=np.float64))
            DATABASEy.append(np.array([bool(last_line.split(',')[0].lower() =="true"),float(last_line.split(',')[1])]))
DATABASEy = np.array(DATABASEy)
DATABASEx = np.array(DATABASEx)


design_space_reduced = DesignSpace(
          [
              IntegerVariable(2, 5),# Marche entre 1 et 8
              FloatVariable(0.01, 1.0), #Density
              FloatVariable(0.0, 1.0), # intolerance
              IntegerVariable(10, 40), #size
              IntegerVariable(1, 10), #vision
              
          ]
      )

# --- 1. Data Preparation ---
# Ensure inputs are the correct shape and type
X = DATABASEx
# Convert boolean Y to integer (False=0, True=1) for scikit-learn
y = DATABASEy[:,0].astype(int) 

X_train = X
X_test = X
y_train = y
y_test = y
print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}") # Check balance between True/False



# --- 3. Preprocessing (Crucial for MLPs) ---
# Neural networks converge faster and better on scaled data
scaler = StandardScaler()
# Fit on training set only, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 4. Define the MLP ---
# hidden_layer_sizes=(100, 50): Two layers, 100 neurons in first, 50 in second
# activation='relu': Rectified Linear Unit (standard for deep learning)
# solver='adam': Stochastic gradient-based optimizer
# max_iter=1000: Increase if the model doesn't converge
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50,25),
    activation='relu',
    solver='adam',
    alpha=0.001, # L2 penalty (regularization) term
    batch_size='auto',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
)

# --- 5. Train ---
print("Training MLP...")
mlp.fit(X_train_scaled, y_train)

mlp2 = MLPRegressor(
    hidden_layer_sizes=(100, 50,25),
    activation='relu',
    solver='adam',
    alpha=0.001, # L2 penalty (regularization) term
    batch_size='auto',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
)

# --- 5. Train ---
print("Training MLP...")
mlp2.fit(X_train_scaled, DATABASEy[:,1] )

# --- 6. Evaluate ---
# Predict on the test set
y_pred = mlp.predict(X_test_scaled)

print("\n--- Performance Report ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['False', 'True']))


start_n = 30
design_space_reduced.seed = 42
samp = MixedIntegerSamplingMethod(LHS, design_space_reduced, criterion="ese", seed=design_space_reduced.seed)
x_doe, _ = samp(start_n, return_is_acting=True)

x_doe_scaled = scaler.transform(x_doe)
y_doe =  mlp.predict(x_doe_scaled)

sampling = MixedIntegerSamplingMethod(
    LHS,
    design_space_reduced,
    criterion="ese",
)
_sampling = lambda n: sampling(n)

    

total_loops = 70

for loop in range(total_loops):
    print(f"LOOP {loop + 1}/{total_loops}")
    
    # --- 0. Calculate Dynamic Alpha ---
    # Alpha starts at 0.0 (Pure Uncertainty Exploration) 
    # and linearly grows to 1.0 (Pure Explanatory Novelty)
    alpha = 0.90
    print(f"--> Hybrid Alpha: {alpha:.2f} (Exploration vs Novelty)")

    # --- 1. Prepare Data for Interpretation ---
    # The primary surrogate model predicting the ABM physics
    sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space_reduced)
    sm.set_training_values(x_doe, y_doe)
    sm.train()
        
    samp = MixedIntegerSamplingMethod(LHS, design_space_reduced, criterion="ese")
    x_val, _ = samp(20, return_is_acting=True)
    x_val1 = np.copy(x_val)
    x_val = np.concatenate((x_val, x_doe))
    
    # We explain the prediction function 
    explainer = shap.ExactExplainer(sm.predict_values, x_doe)
    explanation_object = explainer(x_val)
    shap_values = explanation_object.values
    
    print() # Buffer line to protect console output
    
    n_samples, n_features = shap_values.shape
    
    # Calculate SHAP Densities
    density_matrix = np.zeros((n_features, n_samples))
    for l in range(n_features):
        feature_col = shap_values[:, l]
        kde = gaussian_kde(feature_col, bw_method=0.15)
        densities = kde(feature_col)
        if densities.max() > 0:
            densities = densities / densities.max()
        density_matrix[l, :] = densities
        
    density_mean_matrix = np.mean(density_matrix.T, axis=1)
    y_min_observed = np.min(density_mean_matrix)
            
    density_mean_matrix1 = density_mean_matrix[:20]
    density_mean_matrix2 = density_mean_matrix[20:]
    
    # The Multi-Fidelity surrogate predicting the SHAP density
    surro = MFK(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-9, design_space=design_space_reduced)
    surro.set_training_values(x_val1, density_mean_matrix1, name=0)
    surro.set_training_values(x_doe, density_mean_matrix2)
    surro.train()
    
    # ========================================================
    # --- 2. HYBRID ACQUISITION FUNCTION SETUP ---
    # ========================================================
    # Generate starting points for the optimizer
    x_start = _sampling(15)
    
    # To make Alpha meaningful, we must normalize EI and Sigma 
    # to be on the same scale (0 to 1). We estimate the max values 
    # of the landscape using our x_start samples.
    ei_samples = explicit_ei(np.atleast_2d(x_start), surro, y_min_observed)
    var_samples = sm.predict_variances(np.atleast_2d(x_start))
    sigma_samples = np.sqrt(np.maximum(var_samples, 1e-9))
    
    max_ei = np.max(ei_samples) if np.max(ei_samples) > 1e-12 else 1.0
    max_sigma = np.max(sigma_samples) if np.max(sigma_samples) > 1e-12 else 1.0

    # Define the Hybrid Objective Function for scipy (Negative to MAXIMIZE)
    def obj_fun(x):
        x_2d = np.atleast_2d(x)
        
        # A. Explanatory Novelty (Expected Improvement of Density)
        ei_raw = explicit_ei(x_2d, surro, y_min_observed)[0]
        ei_norm = ei_raw / max_ei # Scale to roughly 0-1
        
        # B. Model Uncertainty (Standard Deviation of Primary Surrogate)
        var_raw = sm.predict_variances(x_2d)[0][0]
        sigma_raw = np.sqrt(max(var_raw, 1e-9))
        sigma_norm = sigma_raw / max_sigma # Scale to roughly 0-1
        
        # C. Combined Hybrid Score
        hybrid_score = (alpha * ei_norm) + ((1 - alpha) * sigma_norm)
        
        # Return negative because scipy minimizes
        return -hybrid_score
    # ========================================================

    method = "SLSQP"
    bounds = design_space_reduced.get_num_bounds()
    cons = []
    for j in range(len(bounds)):
        lower, upper = bounds[j]
        lo = {"type": "ineq", "fun": lambda x, lb=lower, i=j: x[i] - lb}
        up = {"type": "ineq", "fun": lambda x, ub=upper, i=j: ub - x[i]}
        cons.append(lo)
        cons.append(up)
    
    options = {"maxiter": 300}
    opt_all = []
    
    # Run the bounded minimization
    for i in range(len(x_start)):
        opt_all.append(
            minimize(
                lambda x: float(obj_fun(x)),
                x_start[i],
                method=method,
                constraints=cons,
                options=options,
            )
        )        
    
    opt_all = np.asarray(opt_all)
    for opt_i in opt_all:
        if opt_i["message"] == "Maximum number of function evaluations has been exceeded.":
            opt_i["success"] = True
            
    opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
    
    # Find the minimum of our negative hybrid score (which means max utility!)
    obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
    ind_min = np.argmin(obj_success)
    opt = opt_success[ind_min]
    x_opt = np.atleast_2d(opt["x"])
    
    # --- 3. Update Datasets ---
    x_doe = np.concatenate((x_doe, x_opt))
    x_doe_scaled = scaler.transform(x_doe)
    
    y_doe = mlp.predict(x_doe_scaled)
    y2_doe = mlp2.predict(x_doe_scaled)

    DATABASEx = x_doe
    DATABASEy = (np.concatenate((np.atleast_2d(y_doe), np.atleast_2d(y2_doe)), axis=0).T)

    
   

y2_doe =  mlp2.predict(x_doe_scaled)

DATABASEx= x_doe
DATABASEy = (np.concatenate((np.atleast_2d(y_doe),np.atleast_2d(y2_doe)),axis=0).T)
# Extract values where first column is True
true_values = DATABASEy[DATABASEy[:, 0] == True][:, 1]

# Extract values where first column is False
false_values = DATABASEy[DATABASEy[:, 0] == False][:, 1]

def save_database_to_csv(X, y, feature_names, filename="active_learning_results.csv"):
    """
    Saves the design space (X) and the targets (y) to a single CSV file.
    """
    # Create DataFrames
    df_features = pd.DataFrame(X, columns=feature_names)
    df_targets = pd.DataFrame({
        'Convergence': y[:, 0].astype(bool), 
        'Sparsity': y[:, 1].astype(float)
    })
    
    # Concatenate side-by-side
    df_combined = pd.concat([df_features, df_targets], axis=1)
    
    # Save to CSV (index=False prevents pandas from writing row numbers)
    df_combined.to_csv(filename, index=False)
    print(f"--> Database successfully saved to {filename}")
    

# ... (rest of your optimization loop) ...
x_opt = np.atleast_2d(opt["x"])

# 5. Update Datasets
x_doe = np.concatenate((x_doe, x_opt))
x_doe_scaled = scaler.transform(x_doe)

# If you were querying the real simulator, you would evaluate it here.
# Since we are using the MLP proxy:
y_doe_new = mlp.predict_proba(x_doe_scaled)[:, 1] # Fixed probability issue
y2_doe_new = mlp2.predict(x_doe_scaled)

# Update global database trackers
DATABASEx = x_doe
DATABASEy = np.concatenate((np.atleast_2d(y_doe_new), np.atleast_2d(y2_doe_new)), axis=0).T
feature_names = ["Number of Types", "Density", "Intolerance threshold", "Map size", "Perception distance"]

# --- CHECKPOINT SAVE ---
# Save every iteration so you never lose data if it crashes
save_database_to_csv(DATABASEx, DATABASEy, feature_names, filename=f"../data/processed/al_results_loop_{loop}.csv")
# -----------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import warnings

warnings.filterwarnings("ignore") # Suppress seaborn/matplotlib warnings during animation

# ==========================================
# 1. DATA PREPARATION
# ==========================================

# --- YOUR DATA DEFINITION ---
# Assuming x_doe, y_doe, and y2_doe are already defined in your environment.
DATABASEx = x_doe
DATABASEy = (np.concatenate((np.atleast_2d(y_doe), np.atleast_2d(y2_doe)), axis=0).T)

feature_names = ["Number of Types", "Density", "Intolerance threshold", "Map size", "Perception distance"]
total_points = DATABASEx.shape[0]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import warnings

warnings.filterwarnings("ignore") # Suppress seaborn/matplotlib warnings during animation

# ==========================================
# 1. DATA PREPARATION
# ==========================================

# --- YOUR DATA DEFINITION ---
# Assuming x_doe, y_doe, and y2_doe are already defined in your environment.
DATABASEx = x_doe
DATABASEy = (np.concatenate((np.atleast_2d(y_doe), np.atleast_2d(y2_doe)), axis=0).T)

feature_names = ["Number of Types", "Density", "Intolerance threshold", "Map size", "Perception distance"]
total_points = DATABASEx.shape[0]

# ==========================================
# 2. ANIMATION SETUP
# ==========================================

# ==========================================
# 2. ANIMATION SETUP
# ==========================================

# --- FORCE ARRAY SYNC TO FIX MEMORY MISMATCH ---
min_len = min(len(DATABASEx), len(DATABASEy))
DATABASEx = DATABASEx[:min_len]
DATABASEy = DATABASEy[:min_len]
total_points = min_len  # Update total points to the safe synced length
# -----------------------------------------------

# Create the figure ONCE
fig = plt.subplots(figsize=(20, 10))[0]

# Now frames_range is guaranteed to be safe
frames_range = range(start_n, total_points + 1)

# Determine global min/max for features to keep histogram x-axes stable
feature_bounds = [(DATABASEx[:, i].min(), DATABASEx[:, i].max()) for i in range(DATABASEx.shape[1])]

# ==========================================
# 3. THE UPDATE FUNCTION
# ==========================================
def update(N):
    """
    This function is called for every frame.
    N represents the number of points to include in the plot for this frame.
    """
    # --- A. CLEANUP ---
    # Clear the entire figure to safely redraw secondary axes (twinx) without overlapping ghost axes
    fig.clf()
    
    # Add a dynamic title
    fig.suptitle(f"Evolution of Sparsity vs Features (N={N}/{total_points})", fontsize=16, fontweight='bold')

    # --- B. SLICE DATA ---
    currentX = DATABASEx[:N]
    currentY = DATABASEy[:N]

    # --- C. DATAFRAME CREATION ---
    df_sparsity = pd.DataFrame(currentY[:, 1], columns=['Sparsité'])
    df_convergence = pd.DataFrame(currentY[:, 0], columns=['Convergence'])
    df_features = pd.DataFrame(currentX, columns=feature_names)

    df_convergence["Convergence"] = df_convergence["Convergence"].astype(str).str.lower() == '1.0'
    df_sparsity["Sparsité"] = df_sparsity["Sparsité"].astype(float)

    conv_mask = (df_convergence["Convergence"] == 1).values
    
    df_sparsity1 = df_sparsity[conv_mask]
    df_features1 = df_features[conv_mask]
    
    df_sparsity2 = df_sparsity[~conv_mask]
    df_features2 = df_features[~conv_mask]

    # --- D. PLOTTING SCATTERS & HISTOGRAMS (Subplots 0-4) ---
    for i, column in enumerate(df_features.columns):
        ax = fig.add_subplot(2, 3, i + 1)
        
        # Create a secondary axis for the histogram
        ax_hist = ax.twinx()
        
        # Fix the bins based on the global min/max so they don't shift between frames
        fixed_bins = np.linspace(feature_bounds[i][0], feature_bounds[i][1], 16) # 15 bins
        
        # Plot the Histogram (Sampling Density) in the background
        sns.histplot(
            data=df_features, 
            x=column, 
            ax=ax_hist, 
            bins=fixed_bins, 
            stat="density", 
            edgecolor="white"
        )
        
        # --- HIGHLIGHT THE NEWEST POINT'S BAR IN RED ---
        # Get the value of the most recently added point for this feature
        latest_val = df_features[column].iloc[-1]
        
        # Iterate through the bars (patches) drawn by seaborn
        for patch in ax_hist.patches:
            x_left = patch.get_x()
            x_right = x_left + patch.get_width()
            
            # If the newest point falls inside this bar's range, color it red
            if x_left <= latest_val <= x_right:
                patch.set_facecolor('red')
                patch.set_alpha(0.5)
            else:
                patch.set_facecolor('gray')
                patch.set_alpha(0.2)
        # -----------------------------------------------

        # Formatting for the histogram axis
        ax_hist.set_ylabel("Sampling Density", color="gray", fontsize=10)
        ax_hist.tick_params(axis='y', labelcolor="gray")
        ax_hist.grid(False) # Turn off grid for the secondary axis
        
        # Fix X limits so the histogram doesn't shift wildly during animation
        ax.set_xlim(feature_bounds[i][0] * 0.95, feature_bounds[i][1] * 1.05)

        # 1. Main scatter (Background squares)
        ax.scatter(
            df_features[column], 
            df_sparsity["Sparsité"], 
            c=df_convergence["Convergence"], 
            cmap="RdYlGn", 
            alpha=0.8,
            s=15,
            marker="s"
        )
        
        # 2. Converged points (Green +)
        if not df_features1.empty:
            ax.scatter(
                df_features1[column], 
                df_sparsity1["Sparsité"], 
                c="limegreen", 
                alpha=0.6,
                s=100,
                marker="+"
            )
            
        # 3. Non-Converged points (Coral x)
        if not df_features2.empty:
            ax.scatter(
                df_features2[column], 
                df_sparsity2["Sparsité"], 
                c="coral", 
                alpha=0.6,
                s=50,
                marker="x"
            )
            
        # Ensure the scatter plot is drawn ON TOP of the histogram
        ax.set_zorder(ax_hist.get_zorder() + 1)
        ax.patch.set_visible(False) # Make primary axis background transparent
            
        # Formatting for primary axis
        ax.set_xlabel(column, fontsize=12, fontweight='bold')
        if i == 0 or i == 3:
            ax.set_ylabel("Sparsity", fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)

    # --- E. PLOTTING BOXPLOT (Subplot 5) ---
    ax_box = fig.add_subplot(2, 3, 6)
    
    true_vals = df_sparsity[conv_mask]["Sparsité"].values
    false_vals = df_sparsity[~conv_mask]["Sparsité"].values
    
    if len(true_vals) > 0 or len(false_vals) > 0:
        data_box_list = []
        labels_list = []
        
        if len(true_vals) > 0:
            data_box_list.extend(true_vals)
            labels_list.extend(['Yes'] * len(true_vals))
        if len(false_vals) > 0:
            data_box_list.extend(false_vals)
            labels_list.extend(['No'] * len(false_vals))
            
        data_box = pd.DataFrame({
            'value': data_box_list,
            'Convergence?': labels_list
        })

        sns.boxplot(x='Convergence?', y='value', data=data_box, ax=ax_box, palette="RdYlGn_r")
    
    ax_box.set_xlabel('Convergence?', fontsize=12, fontweight='bold')
    ax_box.set_ylabel('Sparsity', fontsize=12, fontweight='bold')
    ax_box.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

# ==========================================
# 4. EXECUTE ANIMATION
# ==========================================

print(f"Generating animation for {total_points - start_n} frames...")

ani = animation.FuncAnimation(fig, update, frames=frames_range, interval=100, blit=False)

output_file = "doe_evolution_with_density_v2.gif"
ani.save(output_file, writer='pillow', fps=1)

print(f"Done! Saved as {output_file}")


# ==========================================
# 5. SHAP BEESWARM ANIMATION
# ==========================================
import shap
from scipy.stats import gaussian_kde

print("Pre-computing SHAP values for the full dataset...")
# Use the final surrogate model (sm) and the initial DoE as the background reference
explainer = shap.ExactExplainer(sm.predict_values, DATABASEx[:start_n])
shap_values_full = explainer(DATABASEx).values

# --- Pre-compute Global Plot Parameters (so points don't "dance") ---
n_samples, n_features = shap_values_full.shape

# 1. Feature Importance (sort order)
feature_importance = np.abs(shap_values_full).mean(axis=0)
sorted_indices = np.argsort(feature_importance)

# 2. Colors (Normalize feature values to 0-1 for the Red-Blue colormap)
norm_f_vals_full = np.zeros_like(DATABASEx)
for i in range(n_features):
    f_min, f_max = DATABASEx[:, i].min(), DATABASEx[:, i].max()
    if f_max - f_min > 0:
        norm_f_vals_full[:, i] = (DATABASEx[:, i] - f_min) / (f_max - f_min)
    else:
        norm_f_vals_full[:, i] = 0.5

# 3. Y-coordinates (Jitter). Calculate once so points stay in place!
y_coords_full = np.zeros((n_samples, n_features))
for i, idx in enumerate(sorted_indices):
    s_vals = shap_values_full[:, idx]
    nbins = 50
    hist, bin_edges = np.histogram(s_vals, bins=nbins)
    bin_indices = np.digitize(s_vals, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, nbins-1)

    jitter = np.zeros_like(s_vals)
    for bin_i in range(nbins):
        count = hist[bin_i]
        if count > 1:
            mask = (bin_indices == bin_i)
            spread_width = min(0.4, 0.02 * count) 
            # Use fixed seed so jitter is perfectly reproducible
            np.random.seed(42 + bin_i + idx) 
            jitter[mask] = np.random.uniform(-spread_width, spread_width, size=count)

    y_coords_full[:, idx] = i + jitter

# --- Animation Setup ---
fig_shap, ax_shap = plt.subplots(figsize=(12, 8))
cmap = plt.get_cmap('coolwarm')
frames_range_shap = range(start_n, n_samples)

def update_beeswarm(N):
    ax_shap.clear()

    # Title showing progress
    fig_shap.suptitle(
        f"Active Learning on SHAP Beeswarm (N={N+1}/{n_samples})\n"
        f"The star marks the exact explanation targeted by the algorithm", 
        fontsize=14, fontweight='bold'
    )

    y_ticks = []
    y_labels = []

    for i, idx in enumerate(sorted_indices):
        # 1. Plot the "old" points (from 0 to N-1)
        ax_shap.scatter(
            shap_values_full[:N, idx], 
            y_coords_full[:N, idx], 
            c=norm_f_vals_full[:N, idx], 
            cmap=cmap, vmin=0, vmax=1,
            s=25, alpha=0.5, edgecolors='none'
        )

        # 2. HIGHLIGHT the "newest" point (N)
        # We plot it as a large star with a black outline so it violently pops out
        ax_shap.scatter(
            shap_values_full[N, idx], 
            y_coords_full[N, idx], 
            c=norm_f_vals_full[N:N+1, idx], # keep as sequence for c
            cmap=cmap, vmin=0, vmax=1,
            s=250, alpha=1.0, marker='*', edgecolors='black', linewidths=1.5, zorder=10
        )

        y_ticks.append(i)
        y_labels.append(feature_names[idx])

    # Formatting
    ax_shap.set_yticks(y_ticks)
    ax_shap.set_yticklabels(y_labels, fontsize=12)
    ax_shap.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax_shap.axvline(x=0, color="#999999", linestyle="-", linewidth=1, zorder=-1)

    # Keep X-axis fixed based on global min/max SHAP values so the screen doesn't zoom in/out
    ax_shap.set_xlim(shap_values_full.min() * 1.1, shap_values_full.max() * 1.1)

    # Clean up spines
    ax_shap.spines['right'].set_visible(False)
    ax_shap.spines['top'].set_visible(False)
    ax_shap.spines['left'].set_visible(False)
    ax_shap.tick_params(axis='y', length=0)

    plt.tight_layout()

# --- Execute Animation ---
print(f"Generating SHAP beeswarm animation for {n_samples - start_n} frames...")
# Slower interval (200ms) so the viewer has time to see where the star lands
ani_shap = animation.FuncAnimation(fig_shap, update_beeswarm, frames=frames_range_shap, interval=200, blit=False)

output_shap_file = "shap_beeswarm_evolution_v2.gif"
ani_shap.save(output_shap_file, writer='pillow', fps=5)

print(f"Done! Saved as {output_shap_file}")

# ==========================================
# 5. UNIFORMITY COMPARISON: AL vs 100-point DoE
# ==========================================
import shap
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

def compute_uniformity(data_matrix, n_bins=15):
    """
    Computes uniformity metrics for points in an N-dimensional space.
    Can be used for both raw Input Features and SHAP explanations.
    """
    n_samples, n_features = data_matrix.shape
    
    # --- Metric 1: Normalized Shannon Entropy ---
    entropies = []
    for i in range(n_features):
        # We add a small offset to the max bin edge to ensure the highest value is included
        hist, _ = np.histogram(data_matrix[:, i], bins=n_bins, density=False)
        hist = hist[hist > 0] # Remove empty bins to avoid log(0) errors
        e = entropy(hist)
        entropies.append(e)
        
    avg_entropy = np.mean(entropies)
    max_entropy = np.log(n_bins)
    normalized_entropy = avg_entropy / max_entropy 
    
    # --- Metric 2: Coefficient of Variation of NND ---
    nn = NearestNeighbors(n_neighbors=2) 
    nn.fit(data_matrix)
    distances, _ = nn.kneighbors(data_matrix)
    nnd = distances[:, 1] 
    
    cv_nnd = np.std(nnd) / np.mean(nnd)
    
    return {"Normalized_Entropy": normalized_entropy, "CV_NND": cv_nnd}

print("\n--- Computing Final Uniformity Comparison ---")

# ==========================================
# 1. Evaluate Pure 100-point LHS DoE Dataset
# ==========================================
print("1/2 Generating pure 100-point LHS DoE...")
samp_100 = MixedIntegerSamplingMethod(LHS, design_space_reduced, criterion="ese", seed=42)
x_doe_100, _ = samp_100(100, return_is_acting=True)

# Calculate INPUT metrics for the LHS DoE
metrics_doe_input = compute_uniformity(x_doe_100)

y_doe_100_prob = mlp.predict_proba(scaler.transform(x_doe_100))[:, 1]

# Train GP on pure DoE
sm_doe = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space_reduced)
sm_doe.set_training_values(x_doe_100, y_doe_100_prob)
sm_doe.train()

# Calculate SHAP metrics for the LHS DoE
explainer_doe = shap.ExactExplainer(sm_doe.predict_values, x_doe_100)
shap_doe_100 = explainer_doe(x_doe_100).values
metrics_doe_shap = compute_uniformity(shap_doe_100)


# ==========================================
# 2. Evaluate Final Active Learning Dataset
# ==========================================
# Ensure we only use the exact budget to be perfectly comparable
DATABASEx = DATABASEx[:100]
x_al_final = DATABASEx

print(f"2/2 Evaluating Active Learning dataset (N={len(x_al_final)})...")

# Calculate INPUT metrics for the Active Learning Dataset
metrics_al_input = compute_uniformity(x_al_final)

# Re-train GP on the final AL dataset to ensure an identical comparison environment
y_al_prob = mlp.predict_proba(scaler.transform(x_al_final))[:, 1]

sm_al = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=design_space_reduced)
sm_al.set_training_values(x_al_final, y_al_prob)
sm_al.train()

# Calculate SHAP metrics for the Active Learning Dataset
explainer_al = shap.ExactExplainer(sm_al.predict_values, x_al_final)
shap_al_final = explainer_al(x_al_final).values
metrics_al_shap = compute_uniformity(shap_al_final)


# ==========================================
# 3. Print Final Summary Table
# ==========================================
print("\n" + "="*95)
print(f"{'':<26}| {'INPUT SPACE (Design Features)':<31} | {'EXPLANATION SPACE (SHAP)':<31}")
print(f"{'Dataset':<26}| {'Entropy (↑)':<14} | {'CV NND (↓)':<14}| {'Entropy (↑)':<14} | {'CV NND (↓)':<14}")
print("-" * 95)
print(f"{'Pure 50-point LHS DoE':<26}| {metrics_doe_input['Normalized_Entropy']:<14.4f} | {metrics_doe_input['CV_NND']:<14.4f}| {metrics_doe_shap['Normalized_Entropy']:<14.4f} | {metrics_doe_shap['CV_NND']:<14.4f}")
print(f"{f'Active Learning (N={len(x_al_final)})':<26}| {metrics_al_input['Normalized_Entropy']:<14.4f} | {metrics_al_input['CV_NND']:<14.4f}| {metrics_al_shap['Normalized_Entropy']:<14.4f} | {metrics_al_shap['CV_NND']:<14.4f}")
print("="*95)
print("↑ Higher Entropy = More uniform distribution across bins.")
print("↓ Lower CV NND   = Points are spread out more evenly without dense, redundant clusters.")



# ==========================================
# 6. VISUALIZE INPUT SPACE SAMPLING (2D Map with Frontier)
# ==========================================
print("\nGenerating 2D Input Space Comparison Map with MLP Frontier...")
from matplotlib.lines import Line2D

# Feature indices based on your design_space definition
idx_density = 1
idx_intolerance = 2

# 1. Get boolean convergence masks (Thresholding probabilities at 0.5)
conv_lhs = y_doe_100_prob >= 0.5
conv_al = DATABASEy[:100, 0] >= 0.5

# Map boolean values to colors (Green = Converged, Red = Failed)
colors_lhs = np.where(conv_lhs, 'limegreen', 'coral')
colors_al = np.where(conv_al, 'limegreen', 'coral')

fig_map, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# --- NEW: CALCULATE THE 5D MLP DECISION FRONTIER ---
# Create a 2D grid of Density and Intolerance values
d_min, d_max = DATABASEx[:, idx_density].min(), DATABASEx[:, idx_density].max()
i_min, i_max = DATABASEx[:, idx_intolerance].min(), DATABASEx[:, idx_intolerance].max()
dd, ii = np.meshgrid(np.linspace(d_min-0.25, d_max+0.25, 200), np.linspace(i_min-0.25, i_max+0.25, 200))

# To query the 5D MLP, we must freeze the other 3 features at their mean values
mean_features = np.mean(DATABASEx, axis=0)
mesh_points = np.zeros((dd.size, 5))

for j in range(5):
    if j == idx_density:
        mesh_points[:, j] = dd.ravel()
    elif j == idx_intolerance:
        mesh_points[:, j] = ii.ravel()
    else:
        mesh_points[:, j] = mean_features[j] # Freeze at the mean

# Scale and predict the grid
mesh_probs = mlp.predict_proba(scaler.transform(mesh_points))[:, 1]
mesh_probs = mesh_probs.reshape(dd.shape)
# ---------------------------------------------------

for ax in axes:
    # Plot the soft probability background (Red -> Green)
    contour = ax.contourf(dd, ii, mesh_probs, levels=50, cmap='RdYlGn', alpha=0.3)
    
    # Draw a hard black dashed line exactly at the 0.5 tipping point
    ax.contour(dd, ii, mesh_probs, levels=[0.5], colors='black', linestyles='--', linewidths=2)

# --- Subplot 1: Pure LHS DoE ---
axes[0].scatter(
    x_doe_100[:, idx_density], 
    x_doe_100[:, idx_intolerance], 
    c=colors_lhs, alpha=0.9, edgecolors='k', s=60, zorder=5
)
axes[0].set_title(f"Pure LHS DoE (N={len(x_doe_100)})", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Density", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Intolerance threshold", fontsize=12, fontweight='bold')

# Create a custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Converged (Yes)', markerfacecolor='limegreen', markeredgecolor='k', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Gridlocked (No)', markerfacecolor='coral', markeredgecolor='k', markersize=10),
    Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='MLP Decision Boundary (50%)')
]
axes[0].legend(handles=legend_elements, loc='upper left', framealpha=0.9)

# --- Subplot 2: Active Learning ---
axes[1].scatter(
    x_al_final[:, idx_density], 
    x_al_final[:, idx_intolerance], 
    c=colors_al, alpha=0.9, edgecolors='k', s=60, zorder=5
)
axes[1].set_title(f"Active Learning (N={len(x_al_final)})", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Density", fontsize=12, fontweight='bold')
axes[1].legend(handles=legend_elements, loc='upper left', framealpha=0.9)

# Formatting
fig_map.suptitle("Input Space Exploration vs. MLP Decision Boundary\n(Other 3 parameters held at their mean values)", fontsize=16, fontweight='bold')
plt.tight_layout()

# Save and Show
output_map_file = "../figures/analysis/input_space_comparison_2D_frontier.png"
plt.savefig(output_map_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"Done! Saved as {output_map_file}")



