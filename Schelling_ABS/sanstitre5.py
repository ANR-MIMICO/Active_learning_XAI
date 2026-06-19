import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import io

#from torch_mas.data import DataBuffer
#import torch as torch
import time
import asyncio
from sklearn import tree
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
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import tabpfn
from smt.surrogate_models import KRG, KPLS, LS, QP, RBF,IDW,RMTB,RMTC,GPX,GENN
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
from itertools import product
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
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
# %%
import openturns as ot
import openturns.viewer as viewer
# %%
from matplotlib import pylab as plt
import torch

ot.Log.Show(ot.Log.NONE)

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
ALL = np.concatenate((DATABASEx, DATABASEy), axis=1)


design_space_reduced = DesignSpace(
          [
              IntegerVariable(2, 5),# Marche entre 1 et 8
              FloatVariable(0.01, 1.0), #Density
              FloatVariable(0.0, 1.0), # intolerance
              IntegerVariable(10, 40), #size
              IntegerVariable(1, 10), #vision
              
          ]
      )
  

xdoes = NestedLHS(nlevel=3, design_space=design_space_reduced, random_state=0)
xt_200, xt_100, xt_50 = xdoes(50)  
# print(len(np.array([np.where(np.all(DATABASEx == xt_200[i], axis=1)) for i in range(200)]).flatten()))
matching_rows50 = np.array([np.where(np.all(DATABASEx == xt_50[i], axis=1)) for i in range(50)]).flatten()
matching_rows100 = np.array([np.where(np.all(DATABASEx == xt_100[i], axis=1)) for i in range(100)]).flatten()
# print(len(np.array([np.where(np.all(np.atleast_2d(matching_rows100) == matching_rows50[i], axis=0)) for i in range(250)]).flatten()))

# predictionsns


column = 1


xt = DATABASEx[matching_rows50][:,:]
yt =DATABASEy[matching_rows50][:,column]


#from torch_mas.data import DataBuffer
#import torch as torch
test_size = 1000
xval = DATABASEx
yval =DATABASEy[:,column]

mpl.rcParams['font.size'] = 15       # base font size
mpl.rcParams['axes.titlesize'] = 20   # title
mpl.rcParams['axes.labelsize'] = 18   # x/y labels
mpl.rcParams['xtick.labelsize'] = 16  # tick labels
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 12  # legend text
# Your sample data
X = np.array([
    0.9658, 0.9522, 0.9495, 0.9079, 0.9393, 0.9128, 0.9473,
    0.8763, 0.9095, 0.8541, 0.9079, 0.7585, 0.8201, 0.7596,
    0.7292, 0.8545
])

def empirical_survival(data, grid_points=1000):
    """
    Compute empirical survival function:
    S(x) = # {samples > x} / N
    """
    x = np.linspace(0, 1, grid_points)
    counts = np.sum(data[:, None] > x[None, :], axis=0)
    return x, counts #/ len(data)

# Generate values
x_vals, surv = empirical_survival(X)

# =============================================================================
# # Plot
# plt.figure(figsize=(8,5))
# plt.step(x_vals*100, surv, where='post', linewidth=2)
# plt.xlabel('SHAP agreement between the surrogate models - NDCG (%)')
# plt.ylabel('Count of surrogate models agreeing')
# #plt.title('SHAP NDCG data point agreement')
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("ndcg.jpg")
# plt.show()
# 
# =============================================================================

from sklearn.metrics import ndcg_score

import numpy as np
import glob
import os

# Specify the directory containing .npy files
directory = '.'  # Replace with your directory path

inputval = 0
# Get a list of all .npy files in the directory
file_list = [f for f in glob.glob("*.npy") if "_"+str(inputval) in f]
# Load each .npy file into a list
arrays = [np.load(file) for file in file_list]

from itertools import combinations
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

num_models = len(arrays)
num_points, num_features = arrays[0].shape
pairwise_agreements = []
ndcg = np.eye(num_models)
disagreement_counts = np.zeros((num_points, num_features), dtype=np.float64)
agreement_counts = np.zeros((num_points, num_features), dtype=np.float64)
for i, j in combinations(range(num_models), 2):
    model_i = arrays[i]
    model_j = arrays[j]

    # Compute ranks for each model
    ranks_i = np.argsort(np.argsort(-model_i, axis=0), axis=0)
    ranks_j = np.argsort(np.argsort(-model_j, axis=0), axis=0)

    # Compare ranks
    agreement = np.abs(ranks_i - ranks_j ) < 10  # Shape: (200, 5)
    agreement = np.abs(model_i-model_j)/(1+np.abs(model_i)+np.abs(model_j))
    agreement = np.abs(np.abs(model_i)-np.abs(model_j))
    agreement =  (model_i+model_j)/240
    disagreement = np.abs((model_j)-(model_i))/(np.abs(model_i)+np.abs(model_j)+1e-9)/(240)

#    pairwise_agreements.append( np.array(agreement,dtype=np.float64))
    disagreement_counts += disagreement.astype(np.float64)
    agreement_counts += agreement.astype(np.float64)
    ndcg[i,j] = ndcg_score(np.abs(model_i), np.abs(model_j))
    ndcg[j,i] = ndcg_score(np.abs(model_j), np.abs(model_i))

# =============================================================================
# 
# # Get a list of all .npy files in the directory
# file_list = [f for f in glob.glob("*.npy") if "_0" in f]
# # Load each .npy file into a list
# arrays = [np.load(file) for file in file_list]
# 
# from itertools import combinations
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# 
# num_models = len(arrays)
# num_points, num_features = arrays[0].shape
# pairwise_agreements = []
# disagreement_counts = np.zeros((num_points, num_features), dtype=np.float64)
# agreement_counts = np.zeros((num_points, num_features), dtype=np.float64)
# for i, j in combinations(range(num_models), 2):
#     model_i = arrays[i]
#     model_j = arrays[j]
# 
#     # Compute ranks for each model
#     ranks_i = np.argsort(np.argsort(-model_i, axis=0), axis=0)
#     ranks_j = np.argsort(np.argsort(-model_j, axis=0), axis=0)
# 
#     # Compare ranks
#     agreement = np.abs(ranks_i - ranks_j ) < 10  # Shape: (200, 5)
#     agreement = np.abs(model_i-model_j)/(1+np.abs(model_i)+np.abs(model_j))
#     agreement = np.abs(np.abs(model_i)-np.abs(model_j))
#     agreement =  (model_i+model_j)/128
#     disagreement = np.abs((model_j)-(model_i))/(np.abs(model_i)+np.abs(model_j)+1e-9)/128
# 
# #    pairwise_agreements.append( np.array(agreement,dtype=np.float64))
#     disagreement_counts += disagreement.astype(np.float64)
#     agreement_counts += agreement.astype(np.float64)
#     ndcg[i,j] += ndcg_score(np.abs(model_i), np.abs(model_j))
#     ndcg[j,i] += ndcg_score(np.abs(model_j), np.abs(model_i))
# 
# 
# =============================================================================




shownumber = 4
#plt.scatter(np.unique(xval,axis=0)[:,shownumber],disagreement_counts[:,shownumber])
#plt.show()
#plt.scatter(np.unique(xval,axis=0)[:,shownumber],agreement_counts[:,shownumber])


y = agreement_counts[:, shownumber]
error = disagreement_counts[:, shownumber]
x = np.unique(xval,axis=0)[:, shownumber]



# Get sorting indices based on x
sort_idx = np.argsort(x)

# Apply the same order to all arrays
x = x[sort_idx]
y = y[sort_idx]
error = error[sort_idx]


plt.figure(figsize=(8, 5))

#plt.plot(x, y, color='blue', label='Agreement')
#plt.fill_between(x, y - error, y + error, color='blue', alpha=0.2, label='± Disagreement')
#plt.xlabel('X-axis value')
#plt.ylabel('Agreement Count')
#plt.title('Agreement with Disagreement as Uncertainty')


plt.scatter(x,y)

plt.figure(figsize=(8, 5))
plt.errorbar(
    x, y, yerr=error,
    fmt='o',                # circle markers
    ecolor='gray',          # color of the error bars
    elinewidth=1.5,
    capsize=4,
    markerfacecolor='blue',
    markeredgewidth=0.5,
 #   label='Agreement ± Disagreement'
)
#plt.xlabel('X-axis value')  # customize as needed
#plt.ylabel('Agreement Count')
#plt.title('Agreement with Disagreement as Uncertainty')
#plt.savefig("plot"+str(inputval)+"_"+str(shownumber)+".jpg",dpi=600,bbox_inches='tight')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()




#df_cm = pd.DataFrame(ndcg, index = [i[13:-4] for i in file_list],
#columns = [i[13:-4] for i in file_list])
#plt.figure(figsize = (10,7))
#sns.heatmap(df_cm, annot=True)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Suppose `ndcg` is your 120×120 pairwise matrix, and
# `labels` is your list of 120 strings:
labels = [i[13:-4] for i in file_list]
labels = ['CIEL',
 'DT',
 'GENN',
 'IDW',
 'kNN',
 'GP',
 'LR',
 'MLP',
 'PCE',
 'QP',
 'RBF',
 'RF',
 'RMTS',
 'SVM',
 'TabPFN',
 'GB']

df_cm = pd.DataFrame(np.int32(ndcg*100), index=labels, columns=labels)

# seaborn clustermap
g= sns.clustermap(
    df_cm,
    figsize=(10, 10),
    cmap="viridis",      # or whatever colormap you like
    annot=True,
    fmt=".2f",           # number format in cells
    row_cluster=True,
    col_cluster=True,
    dendrogram_ratio=(0.1, 0.1),
    cbar_pos=(0.02, 0.8, 0.05, 0.18),
)
# 1) Get the reordered leaf indices
row_order = g.dendrogram_row.reordered_ind
col_order = g.dendrogram_col.reordered_ind
mean = list(np.argsort((np.array(row_order)+np.array(col_order))/2))

mean = [5,10, 14, 11, 15, 1, 9, 6, 7, 13, 3, 0, 4, 2,    12, 8   ]
mean = list(np.argsort((np.array(row_order)+np.array(col_order))/2))

# Apply the new order to the DataFrame
ordered_labels = [labels[i] for i in mean]
ordered_labels = ['GP',
 'RBF',
 'TabPFN',
 'RF',
 'GB',
 'QP',
 'LR',
 'MLP',
 'PCE',
 'DT',
 'SVM',
 'IDW',
 'CIEL',
 'kNN',
 'GENN',
 'RMTS',
]

df_cm_ordered = df_cm.loc[ordered_labels, ordered_labels] 
# Plot again with fixed order and no clustering
g.ax_heatmap.set_xticklabels(
    g.ax_heatmap.get_xticklabels(), 
    rotation=45, 
    ha='right',
    fontsize=20  # Adjust as needed
)

g= sns.clustermap(
    df_cm_ordered,
    row_cluster=False,
    col_cluster=True,
    cmap="YlGn",
    cbar= False,
    annot=True,
    cbar_pos=None,
    linewidths=0.25,
    linecolor='black',
    square=True,
    fmt=".0f",
    figsize=(10, 10),
    dendrogram_ratio=(0.1, 0.1),
  #  cbar_pos=(0.02, 0.8, 0.05, 0.18),
)

import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt



n_cols = df_cm_ordered.shape[1]

# 1. Compute standard linkage on the columns
#    (we’ll cluster columns, so pass df_cm_ordered.T)
linkage = sch.linkage(df_cm_ordered.T, method='average')

# 2. The last row in `linkage` is the top‐level merge:
#    linkage.shape[0] == n_cols − 1, so:
root_row_idx = linkage.shape[0] - 1
L_root, R_root = linkage[root_row_idx, 0], linkage[root_row_idx, 1]

#    (L_root and R_root are cluster IDs; if ≥ n_cols, they’re merged clusters,
#     if < n_cols, they’re original columns.)

# 3. We only want to flip the *immediate children* of R_root.
#    In scipy’s linkage array, cluster ID “k” (where k ≥ n_cols)
#    was created at row index (k − n_cols). So:
if R_root < n_cols:
    raise ValueError("The right‐child at the top level was actually a singleton (no subtree to flip).")

target_row = int(R_root - n_cols)

# 4. Copy linkage and swap only that one row’s children
flipped_linkage = linkage.copy()
flipped_linkage[target_row, 0], flipped_linkage[target_row, 1] = (
    flipped_linkage[target_row, 1],
    flipped_linkage[target_row, 0],
)
flipped_linkage[-5, 0], flipped_linkage[-5, 1] = (
    flipped_linkage[-5, 1],
    flipped_linkage[-5, 0],
)
# Plot with custom linkage
g = sns.clustermap(
    df_cm_ordered,
    row_cluster=False,
    col_cluster=True,
    col_linkage=flipped_linkage,  # Use the flipped dendrogram
    cmap="YlGn",
    cbar=False,
    annot=True,
    cbar_pos=None,
    linewidths=0.25,
    linecolor='black',
    square=True,
    fmt=".0f",
    figsize=(10, 10),
    dendrogram_ratio=(0.1, 0.1),
)
#plt.show()


plt.suptitle("Convergence", y=1.02,fontsize=18)
# plt.tight_layout()

#plt.savefig("plot_ndcg_conv2.jpg",dpi=640,bbox_inches='tight')
plt.show()

# =============================================================================
# 
# 
# 
# col_linkage = g.dendrogram_col.linkage
# col_labels = df_cm_ordered.columns.tolist()
# col_order = g.dendrogram_col.reordered_ind
# 
# from scipy.cluster.hierarchy import dendrogram
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(10, 4))
# dendrogram(
#     col_linkage,
#     labels=[col_labels[i] for i in col_order],
#     orientation='top',
#     leaf_rotation=90,
#     leaf_font_size=10,
#     
# )
# #plt.title("Column Dendrogram (Model Clustering)")
# plt.tight_layout()
# 
# plt.savefig("dendoogram_ndcg_spars.jpg",dpi=600)
# plt.show()
# 
# 
# =============================================================================
