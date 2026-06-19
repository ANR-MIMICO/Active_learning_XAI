#!/usr/bin/python
# -*- encoding: utf8 -*-
"""
Created on Fri Nov 15 09:27:23 2024

@author: psaves
"""
import io
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
true_values = DATABASEy[DATABASEy[:, 0] == True][:, 0]
# Extract values where first column is False
false_values = DATABASEy[DATABASEy[:, 0] == False][:, 0]
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



xt = DATABASEx[matching_rows50[matching_rows50 %5 <5]][:,]
yt =DATABASEy[matching_rows50[matching_rows50 %5 <5]][:,0]


# training the model
sm = KRG(use_het_noise=False,poly="linear",corr ="squar_exp",pow_exp_power=1.1, eval_noise=True)
sm.set_training_values(xt, yt)
sm.train()


sm2 = RandomForestRegressor(n_estimators=100, bootstrap=True, oob_score=False)
sm2.fit(xt.reshape(-1, 1) , yt.reshape(-1, 1) )


# predictions
x = np.linspace(0, 1, 1000).reshape(-1, 1)
y = sm.predict_values(x)
y2= np.atleast_2d(sm2.predict(x)).T
var = sm.predict_variances(x)

# Estimate variance using tree predictions
tree_predictions = np.array([tree.predict(x) for tree in sm2.estimators_])  # Get predictions from each tree
rf_var = np.atleast_2d(np.var(tree_predictions, axis=0)).T  # Compute variance


# Define plot size
plt.figure(figsize=(10, 6))


# Training Data Scatter Plot
plt.scatter(xt, yt, label="Training Data", color="black", marker="o", s= 100,  alpha=0.7, edgecolors="white")

# Mean Predictions
plt.plot(x, y2, linewidth=3.5, linestyle="dashed", label="RF Prédiction moyenne", color="blue")

# Plot Random Forest confidence interval
plt.fill_between(
    np.ravel(x),
    np.ravel(y2 - 3 * np.sqrt(rf_var)),
    np.ravel(y2 + 3 * np.sqrt(rf_var)),
    alpha=0.5,
    label="RF 3-sd Confidence Interval",
    color="royalblue"
)

plt.plot(x, y2, linewidth=1.6, linestyle="solid",  color="blue")
plt.plot(x, y, linewidth=3.5, alpha=1.0, linestyle=(0, (3, 1, 1, 1)), label="GP Prédiction moyenne", color="darkorange")
plt.plot(x, y, linewidth=1.6, alpha=1.0, linestyle="solid",  color="darkorange")

# Plot Kriging confidence interval
plt.fill_between(
    np.ravel(x),
    np.ravel(y - 3 * np.sqrt(var)),
    np.ravel(y + 3 * np.sqrt(var)),
    alpha=0.15,
    label="Kriging 3-sd Confidence Interval",
    color="orange"
)

# Formatting
#plt.title("Surrogate models on the density variable", fontsize=14, fontweight="bold")
plt.legend(loc="upper right", fontsize=11, frameon=True)
plt.xlabel(r"$density$", fontsize=12)
plt.ylabel(r"$sparsity$", fontsize=12)
plt.grid(True, linestyle="dotted", alpha=0.6)  # Add a light grid for readability


# Show the plot
plt.savefig("../figures/analysis/1D_models.png", dpi=350, bbox_inches='tight')
plt.tight_layout()
plt.show()



xt = DATABASEx[matching_rows50[matching_rows50 %5 <5]][:,:]
yt =DATABASEy[matching_rows50[matching_rows50 %5 <5]][:,0]


# training the model
sm = KRG(use_het_noise=False,poly="linear",corr ="squar_exp",pow_exp_power=1.1, eval_noise=True)
sm.set_training_values(xt, yt)
sm.train()


sm2 = RandomForestRegressor(n_estimators=100, bootstrap=True, oob_score=False)
sm2.fit(xt, yt.reshape(-1, 1) )

bounds = design_space_reduced.get_num_bounds()
# Correction : Bien encapsuler tous les niveaux dans des listes
levels = [np.linspace(bounds[0, 0]+3, bounds[0, 1], 1)] + \
         [np.linspace(bounds[1, 0], bounds[1, 1], 1000)] + \
         [np.linspace((b[0]+b[1])/2, b[1], 1) for b in bounds[2:]]

levels[0] =[3]
levels[2] =[0.5]
levels[4] =[5]

# Générer la grille par produit cartésien
grid = np.array(list(product(*levels)))


# Trier la grille selon la deuxième colonne (index 1)
grid = grid[grid[:, 1].argsort()]
# predictions
x = np.linspace(0, 1, 200).reshape(-1, 1)


y = sm.predict_values(grid)
y2= np.atleast_2d(sm2.predict(grid)).T
var = sm.predict_variances(grid)

# Estimate variance using tree predictions
tree_predictions = np.array([tree.predict(grid) for tree in sm2.estimators_])  # Get predictions from each tree
rf_var = np.atleast_2d(np.var(tree_predictions, axis=0)).T  # Compute variance

x = np.atleast_2d(grid[:,1]).T

# Define plot size
plt.figure(figsize=(10, 6))


# Training Data Scatter Plot
plt.scatter(np.atleast_2d(xt[:,1]), yt, label="Training data", color="black", marker="o", s= 100,  alpha=0.7, edgecolors="white")

# Mean Predictions
plt.plot(x, y2, linewidth=3.5, linestyle="dashed", label="RF mean prediction", color="blue")

# Plot Random Forest confidence interval
plt.fill_between(
    np.ravel(x),
    np.ravel(y2 - 3 * np.sqrt(rf_var)),
    np.ravel(y2 + 3 * np.sqrt(rf_var)),
    alpha=0.3,
    label="RF 99% Confidence Interval",
    color="royalblue"
)

plt.plot(x, y2, linewidth=1.6, linestyle="solid",  color="blue")
plt.plot(x, y, linewidth=3.5, alpha=1.0, linestyle=(0, (3, 1, 1, 1)), label="GP mean prediction", color="darkorange")
plt.plot(x, y, linewidth=1.6, alpha=1.0, linestyle="solid",  color="darkorange")

# Plot Kriging confidence interval
plt.fill_between(
    np.ravel(x),
    np.ravel(y - 3 * np.sqrt(var)),
    np.ravel(y + 3 * np.sqrt(var)),
    alpha=0.15,
    label="GP 99% Confidence Interval",
    color="orange"
)

# Formatting
#plt.title("Surrogate models on the density variable", fontsize=14, fontweight="bold")
plt.legend(loc="upper right", fontsize=17, frameon=True)
plt.xlabel(r"$Density$", fontsize=17)
plt.ylabel(r"$Sparsity$", fontsize=17)
plt.grid(True, linestyle="dotted", alpha=0.6)  # Add a light grid for readability


# Show the plot
plt.savefig("../figures/analysis/1D_models.png", dpi=350, bbox_inches='tight')
plt.tight_layout()
plt.show()







xt = DATABASEx[matching_rows50][:,:]
yt =DATABASEy[matching_rows50][:,0]


# training the model
sm = KRG(use_het_noise=True,poly="linear",corr ="squar_exp", eval_noise=True,n_start=50)

#sm = LS()
#sm =IDW(p=2)
#sm = RMTC(xlimits=design_space_reduced.get_num_bounds())
#sm = QP()
#sm = RBF(poly_degreee=1)

sm.set_training_values(xt, yt)
sm.train()

# =============================================================================
# 
# dy = np.zeros((5,250))
# for kx in range(5):
#     dy[kx] = sm.predict_derivatives(xt,kx)
# # Instantiate
# sm = GENN()
# # Train
# sm.load_data(xt, yt, dy)
# sm.train()
# 
# =============================================================================
    
#sm =KNeighborsRegressor(n_neighbors=15)
#sm = LinearRegression()

#sm = DecisionTreeClassifier()
#sm = XGBClassifier()  

#sm = LS()


#sm = SVR()  

#sm = MLPClassifier()

#sm.fit(xt, yt)
    
# predictionsns

xval = DATABASEx
yval =DATABASEy[:,0]







y = sm.predict_values(xval)
#y = np.round(sm.predict(xval))

#y= (sm.predict(xval))

#y= np.round(sm.predict(xval))
#var = sm.predict_variances(xval)

print(np.sqrt(1/1000  *np.sum(np.abs(yval-y.T))))

print(np.sum(np.abs(yval-y.T)))


#### Random Forest

import matplotlib.pyplot as plt
import numpy as np

importances = sm.feature_importances_
feature_names = ["Nombre de types", "Densité", "Seuil d'intolérance", "Taille de la carte", "Distance de perception"]
feature_names =["Number of Types", "Density", "Intolerance threshold", "Map size", "Perception distance"]
#feature_names = [" ","  ","   ","    ","     "]
# Plot feature importance
plt.figure(figsize=(8,5))
plt.barh(feature_names, importances*100)
plt.xlabel("Importance (%)", fontsize=16)
#plt.ylabel("Variable")
#plt.title("Importance des Variables dans la prédiction de la convergence")
plt.title("Convergence", fontsize=16)
# Show the plot
plt.gca().invert_yaxis()
plt.savefig("../figures/analysis/imp_RF_conv.png", dpi=350, bbox_inches='tight')
plt.tight_layout()
plt.show()

# =============================================================================
# from sklearn.inspection import permutation_importance
# 
# result = permutation_importance(sm, xt, yt, n_repeats=10, random_state=42)
# 
# plt.figure(figsize=(8,5))
# plt.barh(np.array(feature_names), result.importances_mean)
# plt.xlabel("Permutation Importance")
# plt.ylabel("Features")
# plt.title("Permutation Feature Importance")
# plt.show()
# 
# 
# plt.figure()
# 
# =============================================================================
from sklearn.inspection import PartialDependenceDisplay
feature_names = np.array(["Nb. types", u"Densité", " Intolérance", "Taille de carte", "Perception dist."])
feature_names =["Number of Types", "Density", "Intolerance threshold", "Map size", "Perception distance"]
#feature_names = [" ","  ","   ","    ","     "]

features = [0, 1,2,3,4]  # Indices of features to plot
PartialDependenceDisplay.from_estimator(sm, xt, features, feature_names=feature_names)
plt.show()

plt.figure()
import shap
explainer = shap.TreeExplainer(sm)
shap_values = explainer.shap_values(DATABASEx)
plt.title("Convergence", fontsize=16)

shap.summary_plot(shap_values, DATABASEx, sort=False, show= False,feature_names=feature_names,color_bar_label='Feature value',show_values_in_legend=True )

# Show the plot
plt.savefig("../figures/analysis/shap_RF_conv.png", dpi=350, bbox_inches='tight')
plt.tight_layout()
plt.show()

import shap
plt.title("Convergence", fontsize=16)

explainer = shap.TreeExplainer(sm)
shap_values = explainer.shap_values(DATABASEx)

# For a binary classifier, select the SHAP values for class 1.
shap.summary_plot(shap_values[:,:,1], DATABASEx, show=False, sort=False,
                  feature_names=feature_names,
                  color_bar_label='Valeur de la variable',
                  show_values_in_legend=True)

# Save and show the plot
plt.savefig("../figures/analysis/shap_RF_conv.png", dpi=350, bbox_inches='tight')
plt.tight_layout()
plt.show()

# =============================================================================
# from sklearn.tree import *
# plt.figure()
# 
# tree.plot_tree(sm.estimators_[0], feature_names=feature_names)
# plt.show()
# 
# =============================================================================

