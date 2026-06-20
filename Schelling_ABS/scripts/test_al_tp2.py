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
    # plt.show()

# --- USAGE ---


current_file_path = os.path.abspath(__file__)[-15]
# Specify the directory containing the files
folder_path = os.path.join(os.path.dirname(__file__), '..', 'doe_5_200')

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
            DATABASEx.append(np.array(filename[:-11].split('_'),dtype=np.float64))
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



design_space_reduced.seed = 42
samp = MixedIntegerSamplingMethod(LHS, design_space_reduced, criterion="ese", seed=design_space_reduced.seed)
x_doe, _ = samp(10, return_is_acting=True)

x_doe_scaled = scaler.transform(x_doe)
y_doe =  mlp.predict(x_doe_scaled)

sampling = MixedIntegerSamplingMethod(
    LHS,
    design_space_reduced,
    criterion="ese",
)
_sampling = lambda n: sampling(n)




for loop in range(10) :
    print("LOOP ", loop)
    # --- 1. Prepare Data for Interpretation ---
    # The MLP model was trained on scaled data, so x_val must be scaled using the SAME scaler.
    
    sm = KRG(theta0=[1e-2],print_global=False,eval_noise=True,nugget=1e-8,design_space = design_space_reduced)
    sm.set_training_values(x_doe, y_doe)
    sm.train()
        
    samp = MixedIntegerSamplingMethod(LHS, design_space_reduced, criterion="ese")

    x_val, _ = samp(50, return_is_acting=True)
    x_val1 = np.copy(x_val)
    x_val =np.concatenate((x_val,x_doe))
    
    # We explain the prediction function (predict_proba returns probabilities for [False, True])
    explainer = shap.ExactExplainer(sm.predict_values,x_doe)
    explanation_object = explainer(x_val)

    # Extract the raw numpy array from the explanation object
    shap_values = explanation_object.values
  
                    
    # Parameters
    n_samples, n_features = shap_values.shape
    
    
    density_matrix = np.zeros((n_features, n_samples))
    for l in range(n_features):
        feature_col = shap_values[:, l]
        kde = gaussian_kde(feature_col)
        densities = kde(feature_col)
        if densities.max() > 0:
            densities = densities / densities.max()
        density_matrix[l, :] = densities
 #   density_matrix = (1-density_matrix).T
#    density_matrix = density_matrix/np.exp(np.abs(shap_values))
    density_mean_matrix = np.mean(density_matrix.T,axis=1)
  #  density_mean_matrix = -density_mean_matrix
        
    y_min_observed = np.min(density_mean_matrix)
            
    density_mean_matrix1 = density_mean_matrix[:50]
    density_mean_matrix2 = density_mean_matrix[50:]
    surro =  MFK(theta0=[1e-2],print_global=False,eval_noise=True,nugget=1e-9,design_space = design_space_reduced)
    surro.set_training_values(x_val1, density_mean_matrix1, name=0)
    surro.set_training_values(x_doe, density_mean_matrix2)
    surro.train()
    # Define the objective for scipy (Negative EI)
    def obj_fun(x):
        # SMT models usually expect 2D inputs
        ei = explicit_ei(x, surro, y_min_observed, return_grad=True)
        # Return negative because we want to MAXIMIZE EI
        return -ei[0]
    
    method = "SLSQP"
    options = {"maxiter": 200}
    bounds = design_space_reduced.get_num_bounds()
    cons = []
    for j in range(len(bounds)):
        lower, upper = bounds[j]
        lo = {"type": "ineq", "fun": lambda x, lb=lower, i=j: x[i] - lb}
        up = {"type": "ineq", "fun": lambda x, ub=upper, i=j: ub - x[i]}
        cons.append(lo)
        cons.append(up)
    bounds = None
    options = {"maxiter": 300}
    success = False
    opt_all = []
    x_start = _sampling(15)
    for i in range(len(x_start)):
        opt_all.append(
            minimize(
                lambda x: float(obj_fun(x)),
                x_start[i],
                method=method,
                bounds=bounds,
                constraints=cons,
                options=options,
            )
        )       
    opt_all = np.asarray(opt_all)
    for opt_i in opt_all:
        if (
            opt_i["message"]
            == "Maximum number of function evaluations has been exceeded."
        ):
            opt_i["success"] = True
    opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
    obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
    ind_min = np.argmin(obj_success)
    opt = opt_success[ind_min]
    x_opt = np.atleast_2d(opt["x"])
    
    
    
    # Run optimization
    # Start from a random point or the best point found so far



# =============================================================================
#     criterion = "EI"      
#     ego = EGO(
#         n_iter=1,
#         criterion=criterion,
#         xdoe=x_val,
#         ydoe = -density_mean_matrix,
#         surrogate= KRG(theta0=[1e-2],print_global=False,eval_noise=True,nugget=1e-9,design_space = design_space_reduced),
#     )
#     
# =============================================================================
    
# =============================================================================
# 
#     def function_test_1d(x) : 
#         return np.array([0.5]).reshape((-1, 1))
#     x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=function_test_1d)
#    
# =============================================================================


    x_doe = np.concatenate((x_doe,x_opt))
    x_doe_scaled = scaler.transform(x_doe)
    y_doe =  mlp.predict(x_doe_scaled)


y2_doe =  mlp2.predict(x_doe_scaled)

DATABASEx= x_doe
DATABASEy = (np.concatenate((np.atleast_2d(y_doe),np.atleast_2d(y2_doe)),axis=0).T)
# Extract values where first column is True
true_values = DATABASEy[DATABASEy[:, 0] == True][:, 1]

# Extract values where first column is False
false_values = DATABASEy[DATABASEy[:, 0] == False][:, 1]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# ==========================================
# 1. DATA PREPARATION
# ==========================================

# --- YOUR DATA DEFINITION ---
# Assuming x_doe, y_doe, and y2_doe are already defined in your environment.
# If running this as a standalone script, ensure these loaded before this block.

DATABASEx = x_doe
# Concatenate and Transpose as per your snippet
DATABASEy = (np.concatenate((np.atleast_2d(y_doe), np.atleast_2d(y2_doe)), axis=0).T)

feature_names = ["Number of Types", "Density", "Intolerance threshold", "Map size", "Perception distance"]
total_points = DATABASEx.shape[0]

# ==========================================
# 2. ANIMATION SETUP
# ==========================================

# Create the figure and axes ONCE
fig, axes_grid = plt.subplots(2, 3, figsize=(20, 10))
axes = axes_grid.flatten()

# We will start animating from N=10 points up to the total number of points
start_n = 10
frames_range = range(start_n, total_points + 1)

# ==========================================
# 3. THE UPDATE FUNCTION
# ==========================================
def update(N):
    """
    This function is called for every frame.
    N represents the number of points to include in the plot for this frame.
    """
    
    # --- A. SLICE DATA ---
    # Take only the first N points
    currentX = DATABASEx[:N]
    currentY = DATABASEy[:N]

    # --- B. CLEANUP ---
    # Clear previous plots
    for ax in axes:
        ax.clear()
        
    # Add a dynamic title
    fig.suptitle(f"Evolution of Sparsity vs Features (N={N}/{total_points})", fontsize=16, fontweight='bold')

    # --- C. DATAFRAME CREATION (Local to this frame) ---
    # We recreate the DF every frame based on the sliced data
    df_sparsity = pd.DataFrame(currentY[:, 1], columns=['Sparsité'])
    df_convergence = pd.DataFrame(currentY[:, 0], columns=['Convergence'])
    df_features = pd.DataFrame(currentX, columns=feature_names)

    # Critical: Ensure types are correct (Boolean for color, Float for sparsity)
    # Because numpy concatenation often coerces booleans to floats/objects
    df_convergence["Convergence"] = df_convergence["Convergence"].astype(str).str.lower() == '1.0'
    df_sparsity["Sparsité"] = df_sparsity["Sparsité"].astype(float)

    # Create masks for splitting
    conv_mask = df_convergence["Convergence"] == 1
    
    df_sparsity1 = df_sparsity[conv_mask]
    df_features1 = df_features[conv_mask]
    
    df_sparsity2 = df_sparsity[~conv_mask]
    df_features2 = df_features[~conv_mask]

    # --- D. PLOTTING SCATTERS (Subplots 0-4) ---
    for i, column in enumerate(df_features.columns):
        ax = axes[i]
        
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
            
        # Formatting
        ax.set_xlabel(column, fontsize=12, fontweight='bold')
        if i == 0 or i == 3:
            ax.set_ylabel("Sparsity", fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        
        # Optional: Fix axes limits to the FULL dataset range 
        # (prevents axes from jumping around during animation)
        # ax.set_xlim(DATABASEx[:, i].min(), DATABASEx[:, i].max())
        # ax.set_ylim(DATABASEy[:, 1].astype(float).min(), DATABASEy[:, 1].astype(float).max())

    # --- E. PLOTTING BOXPLOT (Subplot 5) ---
    ax_box = axes[5]
    
    # Combine for Boxplot
    # We use the boolean mask we created earlier
    true_vals = df_sparsity[conv_mask]["Sparsité"].values
    false_vals = df_sparsity[~conv_mask]["Sparsité"].values
    
    # Only plot if we have data to avoid Seaborn errors
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
    # Add padding for the suptitle
    plt.subplots_adjust(top=0.90)

# ==========================================
# 4. EXECUTE ANIMATION
# ==========================================

print(f"Generating animation for {total_points - start_n} frames...")

# Create Animation Object
# interval=100 means 100ms per frame (10 fps)
ani = animation.FuncAnimation(fig, update, frames=frames_range, interval=100, blit=False)

# Save to GIF
output_file = "doe_evolution3.gif"
ani.save(output_file, writer='pillow', fps=10)

print(f"Done! Saved as {output_file}")
# # plt.show() # Uncomment to view in real-time instead of saving
print(np.quantile(DATABASEx[:,2],0.75)-np.quantile(DATABASEx[:,2],0.25))



