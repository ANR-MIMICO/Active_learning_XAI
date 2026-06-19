

indices = [50,
 51,
 52,
 53,
 54,
 190,
 191,
 192,
 193,
 194,
 240,
 241,
 242,
 243,
 244,
 250,
 251,
 252,
 253,
 254,
 370,
 371,
 372,
 373,
 374,
 445,
 446,
 447,
 448,
 449,
 470,
 471,
 472,
 473,
 474,
 500,
 501,
 502,
 503,
 504,
 545,
 546,
 547,
 548,
 549,
 630,
 631,
 632,
 633,
 634,
 665,
 666,
 667,
 668,
 669,
 715,
 716,
 717,
 718,
 719,
 810,
 811,
 812,
 813,
 814,
 895,
 896,
 897,
 898,
 899,
 975,
 976,
 977,
 978,
 979,
 990,
 991,
 992,
 993,
 994]



# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:27:23 2024

@author: psaves
"""
import time
import asyncio
from gama_client.message_types import MessageTypes
from gama_client.sync_client import GamaSyncClient
from typing import Dict
import csv
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
    

# Extract values where first column is True
true_values = DATABASEy[DATABASEy[:, 0] == True][:, 1]

# Extract values where first column is False
false_values = DATABASEy[DATABASEy[:, 0] == False][:, 1]

# Create a dataframe for seaborn boxplot
import pandas as pd

# Create a dataframe with a column for values and a column for the condition
data = pd.DataFrame({
    'value': np.concatenate([true_values, false_values]),
    'condition': ['True'] * len(true_values) + ['False'] * len(false_values)
})

# Plot the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='condition', y='value', data=data)
plt.title('Boxplot of Values Based on Condition (True or False)')
plt.show()
ALL = np.concatenate((DATABASEx, DATABASEy), axis=1)
     

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Creation des DataFrames pour faciliter l'analyse
df_X = pd.DataFrame(DATABASEx, columns=[
    "Nombre de types d'individus",
    "Densite d'individus",
    "Seuil d'intolerance",
    "Taille de la carte",
    "Distance de perception"
])

df_y_binaire = pd.DataFrame(DATABASEy[:, 0], columns=["Convergence"])
df_y_continu = pd.DataFrame(DATABASEy[:, 1], columns=["Sparsite"])

# Tests sur la variable binaire (convergence)

# Calcul de la correlation de Pearson avec la convergence binaire
print("Correlation de Pearson (convergence binaire):")
correlation_bin = df_X.corrwith(df_y_binaire["Convergence"])
print(correlation_bin)

# Test ANOVA pour chaque variable sur la convergence binaire
p_values_bin = []
for i in range(DATABASEx.shape[1]):
    group1 = DATABASEx[DATABASEy[:, 0] == 1, i]
    group0 = DATABASEx[DATABASEy[:, 0] == 0, i]
    _, p = f_oneway(group1, group0)
    p_values_bin.append(p)
print("P-values ANOVA (convergence binaire):", p_values_bin)

# Calcul de l'information mutuelle pour la convergence binaire
mi_bin = mutual_info_classif(DATABASEx, DATABASEy[:, 0])
print("Information mutuelle (convergence binaire):", mi_bin)

# =============================================================================
# # Tracage de scatter plots pour la convergence binaire
# for i in range(DATABASEx.shape[1]):
#     plt.scatter(DATABASEx[:, i], DATABASEy[:, 0], alpha=0.5)
#     plt.xlabel(f"Variable {i+1}")
#     plt.ylabel("Convergence (binaire)")
#     plt.title(f"Variable {i+1} vs Convergence")
#     plt.show()
# 
# =============================================================================

# Tests sur la variable continue (sparsite)

# Calcul de la correlation de Pearson avec la sparsite
print("Correlation de Pearson (sparsite):")
correlation_cont = df_X.corrwith(df_y_continu["Sparsite"])
print(correlation_cont)

# Le test ANOVA n'est pas approprie pour une variable continue.
# On utilisera l'information mutuelle pour la sparsite en mode regression.
mi_cont = mutual_info_regression(DATABASEx, DATABASEy[:, 1])
print("Information mutuelle (sparsite):", mi_cont)

# =============================================================================
# # Tracage de scatter plots pour la sparsite
# for i in range(DATABASEx.shape[1]):
#     plt.scatter(DATABASEx[:, i], DATABASEy[:, 1], alpha=0.5)
#     plt.xlabel(f"Variable {i+1}")
#     plt.ylabel("Sparsite")
#     plt.title(f"Variable {i+1} vs Sparsite")
#     plt.show()
# 
# =============================================================================

import matplotlib.pyplot as plt
import pandas as pd

# Création des DataFrames
df_sparsity = pd.DataFrame(DATABASEy[:, 1], columns=['Sparsité'])
df_convergence = pd.DataFrame(DATABASEy[:, 0], columns=['Convergence'])  # Supposant que la colonne 2 est la convergence
df_features = pd.DataFrame(DATABASEx, columns=["Number of Types", "Density", "Intolerance threshold", "Map size", "Perception distance"])
df_sparsity1 = df_sparsity[df_convergence["Convergence"] == True]
df_features1 = df_features[df_convergence["Convergence"] == True]
df_convergence1 = df_convergence[df_convergence["Convergence"] == True]
df_sparsity2 = df_sparsity[df_convergence["Convergence"] ==False]
df_features2 = df_features[df_convergence["Convergence"] == False]
df_convergence2 = df_convergence[df_convergence["Convergence"] == False]

# Création de la figure avec 5 subplots (1 ligne, 5 colonnes)
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Flatten the 2D axes array
axes = axes.flatten()

# Boucle pour tracer chaque variable contre la sparsité avec coloration
for i, column in enumerate(df_features.columns):
    scatter = axes[i].scatter(
        df_features[column], 
        df_sparsity["Sparsité"], 
        c=df_convergence["Convergence"],  # Couleur en fonction de la convergence
        cmap="RdYlGn",  # Rouge = forte convergence, Vert = faible convergence
        alpha=0.8,
        s=6,
        marker= "s"
    )
    scatter = axes[i].scatter(
        df_features1[column], 
        df_sparsity1["Sparsité"], 
        c="limegreen",  # Couleur en fonction de la convergence
       # cmap="Greens",  # Rouge = forte convergence, Vert = faible convergence
        alpha=0.6,
        s=100,
        marker = "+"
    )
    scatter = axes[i].scatter(
        df_features2[column], 
        df_sparsity2["Sparsité"], 
        c="coral",  # Couleur en fonction de la convergence
    #    cmap="Reds",  # Rouge = forte convergence, Vert = faible convergence
        alpha=0.6,
        s=50,
        marker = "x"
    )
  #  axes[i].scatter(DATABASEx[indices][:,i], DATABASEy[indices][:,1] , c="k", s=200)
        
    
    axes[i].set_xlabel(column)
    if i==0 or i ==3 :
        axes[i].set_ylabel("Sparsity", fontsize=14, fontweight='bold')

    axes[i].set_xlabel(column, fontsize=14, fontweight='bold')  # Labels plus grands
    axes[i].tick_params(axis='both', labelsize=12)  # Agrandir les ticks

# Boxplot in the last subplot (axes[5])
data = pd.DataFrame({
    'value': np.concatenate([true_values, false_values]),
    'Convergence?': ['Yes'] * len(true_values) + ['No'] * len(false_values)
})

sns.boxplot(x='Convergence?', y='value', data=data, ax=axes[5], palette="RdYlGn_r")
axes[5].set_xlabel('Convergence?', fontsize=14, fontweight='bold')
axes[5].set_ylabel('Sparsity', fontsize=14, fontweight='bold')
axes[5].tick_params(axis='both', labelsize=12)

# =============================================================================
    # # Ajouter une barre de couleur
    # cbar = fig.colorbar(scatter, ax=axes[-1])  
    # cbar.set_label("Convergence")
# 
# =============================================================================
plt.savefig("sparsity_vs_features.png", dpi=350, bbox_inches='tight')
plt.tight_layout()
plt.show()
