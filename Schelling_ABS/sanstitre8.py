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
import sys

import matplotlib.pyplot as plt
import numpy as np

from smt.sampling_methods import LHS
from smt.applications.mfk import MFK, NestedLHS
import seaborn as sns


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

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
    k = 5  # Number of neighbors to consider
    n_samples, n_features = shap_vals.shape
    knn_matrix = np.zeros((n_features, n_samples))
    
    for l in range(n_features):
        # Reshape column to (N, 1) as required by sklearn
        feature_col = shap_vals[:, l].reshape(-1, 1)
        
        # Compute KNN
        # n_neighbors = k+1 because the point itself is included as distance 0
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(feature_col)
        distances, _ = nbrs.kneighbors(feature_col)
        
        # We ignore the first column (distance to self, which is 0)
        # We take the average distance to the k actual neighbors
        avg_knn_dist = distances[:, 1:].mean(axis=1)
        
        knn_matrix[l, :] = avg_knn_dist
    knn_matrix= knn_matrix.T
    knn_matrix = knn_matrix/np.exp(np.abs(shap_vals))


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
DATABASEy = np.array(DATABASEy)[:,0]
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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier # Use MLPRegressor if predicting the float value
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. Data Preparation ---
# Ensure inputs are the correct shape and type
X = DATABASEx
# Convert boolean Y to integer (False=0, True=1) for scikit-learn
y = DATABASEy.astype(int) 

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
samp = MixedIntegerSamplingMethod(LHS, design_space_reduced, criterion="ese", random_state=design_space_reduced.seed)
x_val, _ = samp(250, return_is_acting=True)

import shap

# --- 1. Prepare Data for Interpretation ---
# The MLP model was trained on scaled data, so x_val must be scaled using the SAME scaler.
x_val_scaled = scaler.transform(x_val)

from sklearn.inspection import PartialDependenceDisplay
feature_names = np.array(["Nb. types", u"Densité", " Intolérance", "Taille de carte", "Perception dist."])
feature_names =["Number of Types", "Density", "Intolerance threshold", "Map size", "Perception distance"]
#feature_names = [" ","  ","   ","    ","     "]


# --- 2. Initialize SHAP Explainer ---
# KernelExplainer is model-agnostic. It requires a background dataset to simulate "missing" features.
# We use k-means on the training data to create a small, representative background set (faster computation).
background_summary = shap.kmeans(X_train_scaled, 50) 

# We explain the prediction function (predict_proba returns probabilities for [False, True])
explainer = shap.KernelExplainer(mlp.predict_proba,X_train_scaled)

# --- 3. Calculate SHAP Values ---
print("Calculating SHAP values... (This may take a few minutes)")
# We compute SHAP values for the validation set
shap_values = explainer.shap_values(x_val_scaled)[:,:,0]

# --- 4. Plot Beeswarm ---
custom_beeswarm_plot(shap_values, x_val_scaled, feature_names)

                
# Parameters
n_samples, n_features = shap_values.shape
knn_matrix = np.zeros((n_features, n_samples))


density_matrix = np.zeros((n_features, n_samples))
for l in range(n_features):
    feature_col = shap_values[:, l]
    kde = gaussian_kde(feature_col)
    densities = kde(feature_col)
    if densities.max() > 0:
        densities = densities / densities.max()
    density_matrix[l, :] = densities
density_matrix = (1-density_matrix).T
density_matrix = density_matrix/np.exp(np.abs(shap_values))
density_mean_matrix = np.mean(density_matrix,axis=1)

topx= x_val[np.argpartition(density_mean_matrix, -1)[-1:]]



