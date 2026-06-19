# -*- coding: utf-8 -*-
"""
Created on Wed May 28 19:26:14 2025

@author: psaves
"""

import numpy as np
import glob
import os

# Specify the directory containing .npy files
directory = '.'  # Replace with your directory path

# Get a list of all .npy files in the directory
file_list = [f for f in glob.glob("*.npy") if "_0" in f]
# Load each .npy file into a list
arrays = [np.load(file) for file in file_list]

from itertools import combinations

num_models = len(arrays)
num_points, num_features = arrays[0].shape
pairwise_agreements = []
agreement_counts = np.zeros((num_points, num_features), dtype=int)
for i, j in combinations(range(num_models), 2):
    model_i = arrays[i]
    model_j = arrays[j]

    # Compute ranks for each model
    ranks_i = np.argsort(np.argsort(-model_i, axis=0), axis=0)
    ranks_j = np.argsort(np.argsort(-model_j, axis=0), axis=0)

    # Compare ranks
    agreement = np.abs(ranks_i - ranks_j ) < 15  # Shape: (200, 5)
    pairwise_agreements.append( np.array(agreement,dtype=np.float64))
    agreement_counts += agreement.astype(int)
    