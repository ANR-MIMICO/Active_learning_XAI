# -*- coding: utf-8 -*-
"""
Created on Sat May 10 17:00:30 2025

@author: psaves
"""

#!/usr/bin/python
# -*- encoding: utf8 -*-
"""
Created on Fri Nov 15 09:27:23 2024

@author: psaves
"""
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef as mcc
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

#yt[yt==0] = -1

#sm = KRG(use_het_noise=True,poly="linear",corr ="squar_exp", eval_noise=True,n_start=50)
#sm= KRG()
#sm = LS()
#sm = QP()
#sm = RBF(poly_degree=1)
#sm = IDW(p=1)  
#sm = RMTC(xlimits=design_space_reduced.get_num_bounds())
#from torch_mas.data import DataBuffer
#import torch as torch



test_size = 1000
num_features = 5

# =============================================================================
# 
# sm = tabpfn.TabPFNRegressor()
# sm = tabpfn.TabPFNClassifier(n_jobs=-1, device="cpu",fit_mode="fit_with_cache") 
# sm =  RandomForestClassifier()
# sm =  DecisionTreeClassifier()
# sm =KNeighborsClassifier(n_neighbors=15)
# sm = SVC()
# sm = XGBClassifier()
# sm.fit(xt,yt)
# 
# =============================================================================

sm = LS()

sm.set_training_values(xt, yt)
sm.train()

# =============================================================================
# dy = np.zeros((5,250))
# for kx in range(5):
#     dy[kx] = sm.predict_derivatives(xt,kx)
# # Instantiate
# sm = GENN(is_normalize=True,num_iterations=1250,num_epochs=2,hidden_layer_sizes=[20,20])
# # Train
# sm.load_data(xt, yt, dy)
# sm.train()
# 
# =============================================================================
    


#dataset = DataBuffer(torch.from_numpy(xt).float(), torch.from_numpy(yt).float().unsqueeze(-1))


# =============================================================================
# 
# import time
# from torch_mas.sequential.trainer import BaseTrainer as Trainer
# from torch_mas.sequential.internal_model import LinearWithMemory
# from torch_mas.sequential.activation_function import BaseActivation
# 
# 
# validity = BaseActivation(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     alpha=0.1, 
# )
# internal_model = LinearWithMemory(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     l1=0.1, 
#     memory_length=10, 
# )
# model = Trainer(
#     validity,
#     internal_model,
#     R=0.1,
#     imprecise_th=0.01,
#     bad_th=0.1,
#     n_epochs=5,
# )
#  
# t = time.time() 
# model.fit(dataset)
# tt = time.time() - t
# print(f"Total training time: {tt}s")
#  
# print("Number of agents created:", model.n_agents)
#  
#  
# =============================================================================


# =============================================================================
# 
# import time
# from torch_mas.sequential.trainer import ClassifTrainer as Trainer
# from torch_mas.sequential.internal_model import SVM
# from torch_mas.sequential.activation_function import BaseActivation
# 
# 
# validity = BaseActivation(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     alpha=5, 
# )
# internal_model = SVM(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     l1=0.1, 
#     memory_length=50, 
# )
# model = Trainer(
#     validity,
#     internal_model,
#     R=0.3,
#     imprecise_th=0.0015,
#     bad_th=0.1,
#     n_epochs=50,
# )
# 
# t = time.time()
# model.fit(dataset)
# tt = time.time() - t
# print(f"Total training time: {tt}s")
# print("Number of agents created:", model.n_agents)
# 
# 
# 
# =============================================================================


# =============================================================================
# 
# ot.ResourceMap.SetAsUnsignedInteger("FittingTest-LillieforsMaximumSamplingSize", 1000)
# ot.ResourceMap.SetAsUnsignedInteger(
#         "FunctionalChaosAlgorithm-MaximumTotalDegree", 10
#     )
# 
# 
# xt_ot = ot.Sample(xt.tolist())
# yt_ot = ot.Sample([[y] for y in yt])  # yt doit être 2D !
# sm = ot.FunctionalChaosAlgorithm(xt_ot, yt_ot)
# sm.run()
# result = sm.getResult()
# sm =  result.getMetaModel()
# =============================================================================


#sm = RMTC(xlimits=design_space_reduced.get_num_bounds())
#sm = IDW(xlimits=design_space_reduced.get_num_bounds())
#sm = IDW(p=2)
#sm = KRG(use_het_noise=True,poly="linear",corr ="squar_exp", eval_noise=True,n_start=50)
#sm = QP()
#sm = LS()
#sm.set_training_values(xt,yt)
#sm.train()
#sm = MLPRegressor()
#sm = SVR()  
#sm =KNeighborsRegressor(n_neighbors=15)
#sm = DecisionTreeRegressor()
#sm = XGBRegressor()

#sm = tabpfn.TabPFNRegressor()
#sm.fit(xt,yt)

#sm.set_training_values(xt, yt)
#sm.train()

xval = DATABASEx
yval =DATABASEy[:,column]

# =============================================================================
# Nb = 1000  # Number of data points
# import torch as torch 
# xval2 = torch.from_numpy(xval)
#   
# =============================================================================
y=  sm.predict_values(xval)
#y = model.predict(xval2.float())
#y= y.detach().cpu().numpy()[:,0]
#yold= y
#yval[yval==0] = -1
#y = np.array(sm(ot.Sample(xval.tolist())))[:,0]
y = np.round( y *(y>0))
y[y>1] =1
#y = (sm.predict(xval))

#y= (sm.predict(xval))

#y= np.round(sm.predict(xval))

print(np.sqrt(1/1000  *np.sum(np.abs(yval-y.T))))

print(np.sqrt(1/1000  *np.sum(np.abs(yval-y.T)**2)))

print("classif error: ", np.sum(np.abs(yval-y.T)))
#print ("MCC: ",mcc(yval, y))

print(sm)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming yval = true labels, y_pred = predicted labels (0 or 1)
y_pred = (y >= 0.5).astype(int)  # example thresholding
# =============================================================================
# 
# # Compute confusion matrix
# cm = confusion_matrix(yval, y_pred)
# 
# # Display confusion matrix with labels
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
# disp.plot(cmap=plt.cm.Blues)  # You can choose different colormaps
# plt.title("Confusion Matrix lr")
# plt.show()
# 
# =============================================================================


from smt_explainability.shap import compute_shap_values

# =============================================================================
# 
# shap_values = compute_shap_values(
#     sm,
#     xval,
#     xt,
#     [False]*5,
#     method="exact",
# )
# =============================================================================

#sm = tabpfn.TabPFNRegressor(n_jobs=-1, device="cpu",fit_mode="fit_with_cache")
sm = RandomForestRegressor()
sm =KNeighborsRegressor(n_neighbors=15)
sm= XGBRegressor()
#sm = KRG(use_het_noise=True,poly="linear",corr ="squar_exp", eval_noise=True,n_start=50)
#sm = IDW(p=1)
#sm = RBF(poly_degree=1)
#sm = RMTC(xlimits=design_space_reduced.get_num_bounds())

# training the model
#sm = KRG(use_het_noise=True,poly="linear",corr ="squar_exp", eval_noise=True,n_start=50)

#sm = LS()
#sm =IDW(p=2)
#sm = RMTC(xlimits=design_space_reduced.get_num_bounds())
#sm = QP()
#sm = RBF(poly_degreee=1)

# =============================================================================
# sm.set_training_values(xt, yt)
# sm.train()
# 

# =============================================================================
# ot.ResourceMap.SetAsUnsignedInteger("FittingTest-LillieforsMaximumSamplingSize", 1000)
# ot.ResourceMap.SetAsUnsignedInteger(
#         "FunctionalChaosAlgorithm-MaximumTotalDegree", 10
#     )
# 
# 
# xt_ot = ot.Sample(xt.tolist())
# yt_ot = ot.Sample([[y] for y in yt])  # yt doit être 2D !
# sm = ot.FunctionalChaosAlgorithm(xt_ot, yt_ot)
# sm.run()
# result = sm.getResult()
# sm =  result.getMetaModel()
# 
# =============================================================================

# =============================================================================
# 
# import time
# from torch_mas.sequential.trainer import BaseTrainer as Trainer
# from torch_mas.sequential.internal_model import LinearWithMemory
# from torch_mas.sequential.activation_function import BaseActivation
# 
#  
# dataset = DataBuffer(torch.from_numpy(xt).float(), torch.from_numpy(yt).float().unsqueeze(-1))
# 
# validity = BaseActivation(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     alpha=0.1, 
# )
# internal_model = LinearWithMemory(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     l1=0.1, 
#     memory_length=10, 
# )
# sm = Trainer(
#     validity,
#     internal_model,
#     R=0.5,
#     imprecise_th=0.01,
#     bad_th=0.1,
#     n_epochs=5,
# )
# 
# t = time.time() 
# sm.fit(dataset)
# tt = time.time() - t
# print(f"Total training time: {tt}s")
#   
# print("Number of agents created:", sm.n_agents)
# =============================================================================


# =============================================================================
# import time
# from torch_mas.sequential.trainer import ClassifTrainer as Trainer
# from torch_mas.sequential.internal_model import SVM
# from torch_mas.sequential.activation_function import BaseActivation
# 
# 
# validity = BaseActivation(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     alpha=5, 
# )
# internal_model = SVM(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     l1=0.1, 
#     memory_length=50, 
# )
# model = Trainer(
#     validity,
#     internal_model,
#     R=0.5,
#     imprecise_th=0.01,
#     bad_th=0.1,
#     n_epochs=5,
# )
# 
# t = time.time()
# model.fit(dataset)
# tt = time.time() - t
# print(f"Total training time: {tt}s")
# print("Number of agents created:", model.n_agents)
# 
# sm  = model
# =============================================================================






# =============================================================================
# 
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
    

sm = RBF(poly_degree=1)
sm = IDW(p=2)
sm = RMTC(xlimits=design_space_reduced.get_num_bounds())
sm = KRG(use_het_noise=True,poly="linear",corr ="squar_exp", eval_noise=True,n_start=50)
sm= XGBRegressor()
sm =KNeighborsRegressor(n_neighbors=15)
sm = RandomForestRegressor()
sm = DecisionTreeRegressor()
sm = SVR()  
sm = MLPRegressor()
sm = tabpfn.TabPFNRegressor(n_jobs=-1, device="cpu",fit_mode="fit_with_cache")
sm = DecisionTreeClassifier()
sm = IDW(p=2)
sm =KNeighborsRegressor(n_neighbors=15)
sm = IDW(p=2)
sm = KRG(use_het_noise=True,poly="linear",corr ="abs_exp", eval_noise=True,n_start=50)
sm= RandomForestRegressor()
sm = XGBRegressor()
sm = SVR()  
sm = KRG(use_het_noise=True,poly="linear",corr ="abs_exp", eval_noise=True,n_start=50)

#sm.fit(xt,yt)
sm.set_training_values(xt, yt)
sm.train()

# =============================================================================
# 
# 
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
    


xval = DATABASEx
yval =DATABASEy[:,column]
shap_values = compute_shap_values(
    sm,
    np.unique(xval,axis=0),
    np.unique(xt,axis=0),
    [False]*5,
    method="exact",
)   

#np.save( "./shap_values_"+str(column)+"dt"+".npy",shap_values)

a = np.load("./shap_values_ref_conv.npy")

import shap
plt.title("sparsity", fontsize=16)
feature_names =["Number of Types", "Density", "Intolerance threshold", "Map size", "Perception distance"]

shap.summary_plot(shap_values, np.unique(xval,axis=0), sort=False, show= False,feature_names=feature_names,color_bar_label='Feature value',show_values_in_legend=True )

# Show the plot
plt.tight_layout()
plt.show()


from sklearn.metrics import ndcg_score

ndcg = ndcg_score(np.abs(a), np.abs(shap_values))

print("ndcg= ", ndcg)
# =============================================================================
# tree_predictions = np.array([tree.predict(xval) for ee in sm.estimators_])  # Get predictions from each tree
# 
# #variance = sm.predict_variances(xval)  # Predicted variances (N x 1 array)
# variance = np.atleast_2d(np.var(tree_predictions, axis=0)).T  # Compute variance
# 
# =============================================================================

#s2 = np.abs(np.array(sm.predict(xval,output_type="quantiles",quantiles=[0.15865,0.84135]))-y)
#variance = (np.sum(s2,axis=0)/2)**2

# =============================================================================
# 
# from sklearn.ensemble import RandomForestRegressor
# from mapie.quantile_regression import MapieQuantileRegressor
# from sklearn.model_selection import train_test_split
# X_model, X_calib, y_model, y_calib = train_test_split(
#     xt, yt,
#     test_size=0.5,       # half goes to calibration
#     random_state=42      # for reproducibility
# )
# 
# =============================================================================

#sm.fit(X_model, y_model)

# =============================================================================
# xt_ot = ot.Sample(X_model.tolist())
# yt_ot = ot.Sample([[y] for y in y_model])  # yt doit être 2D !
# sm = ot.FunctionalChaosAlgoritm(xt_ot, yt_ot)
# sm.run()
# result = sm.getResult()
# sm =  result.getMetaModel()
# 
# =============================================================================

#sm.train()
# =============================================================================
# 
# 
# dataset = DataBuffer(torch.from_numpy(X_model).float(), torch.from_numpy(y_model).float().unsqueeze(-1))
# 
# validity = BaseActivation(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     alpha=0.1, 
# )
# internal_model = LinearWithMemory(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     l1=0.1, 
#     memory_length=10, 
# )
# model = Trainer(
#     validity,
#     internal_model,
#     R=0.5,
#     imprecise_th=0.01,
#     bad_th=0.1,
#     n_epochs=5,
# )
# 
# 
# 
# t = time.time() 
# model.fit(dataset)
# tt = time.time() - t
# print(f"Total training time: {tt}s")
# 
# print("Number of agents created:", model.n_agents)
# 
# 
# 
# 
# # 2b. Predict on the calibration half
# #y_calib_pred = sm.predict(X_calib)
# 
# #y_calib_pred  = np.array(sm(ot.Sample(X_calib.tolist())))[:,0]
# 
# X_calib = torch.from_numpy(X_calib).float()
#   
# #y=  sm.predict_values(xval)
# y = model.predict(X_calib.float())
# y_calib_pred= y.detach().cpu().numpy()[:,0]
# 
# 
# 
# # 3a. Nonconformity scores: absolute residuals
# scores = np.abs(y_calib - y_calib_pred)
# 
# # 3b. Quantile at 1 - alpha
# alpha = 0.3173                   # for ~68% coverage
# #scores = scores - sm.predict_values(X_model)
# # Calibration quantile
# q_alpha = np.quantile(scores, 1 - alpha)
# 
# 
# # =============================================================================
# # xt_ot = ot.Sample(X_calib.tolist())
# # yt_ot = ot.Sample([[y] for y in np.abs(y_calib -y_calib_pred)])  # yt doit être 2D !
# # sm = ot.FunctionalChaosAlgorithm(xt_ot, yt_ot)
# # sm.run()
# # result = sm.getResult()
# # sm =  result.getMetaModel()
# # 
# # =============================================================================
# yt_ot = np.abs(y_calib - y_calib_pred)
# 
# 
# dataset = DataBuffer(X_calib, torch.from_numpy(yt_ot).float().unsqueeze(-1))
# 
# validity = BaseActivation(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     alpha=0.1, 
# )
# internal_model = LinearWithMemory(
#     dataset.input_dim, 
#     dataset.output_dim, 
#     l1=0.1, 
#     memory_length=10, 
# )
# model = Trainer(
#     validity,
#     internal_model,
#     R=0.5,
#     imprecise_th=0.01,
#     bad_th=0.1,
#     n_epochs=5,
# )
# 
# 
# 
# t = time.time() 
# model.fit(dataset)
# tt = time.time() - t
# print(f"Total training time: {tt}s")
# 
# print("Number of agents created:", model.n_agents)
# 
# 
# 
# #sm.fit(X_calib, np.abs(y_calib - sm.predict(X_calib)))
# #sm.train()
# 
# 
# #y_pred = sm.predict(xval)
# #y_pred = np.array(sm(ot.Sample(xval.tolist())))[:,0]
# 
# xval = torch.from_numpy(xval)
# y = model.predict(xval.float())
# y_pred= y.detach().cpu().numpy()[:,0]
# 
# 
# 
# 
# sigma_hat = y_pred  + q_alpha               # calibrated residual bound
# 
# # Uncalibrated intervals:
# # lo_uncalib = y_pred - q
# # hi_uncalib = y_pred + q
# 
# # (Already calibrated by split CP:)
# #lo, hi = y_pred - q, y_pred + q
# 
# # 4. Compute interval half‐width (radius) and σ̂
# 
# variance = sigma_hat**2
# 
# 
# # Calculate squared error normalized by variance
# 
# #variance = sm.predict_variances(xval)
# sqdiff = ((yval - yold.T) ** 2).T
# sqdiff = np.atleast_2d(sqdiff).T
# 
# # perform division only where variance>0, leave others unchanged (e.g. zero)
# error = np.divide(
#     sqdiff.T,
#     np.atleast_2d(variance),
#     out=np.zeros_like(sqdiff.T, dtype=float),
#     where=(variance > 1e-8)
# )
# # Compute PVA with logarithm
# pva = (np.log(np.sum(error) / Nb))
# 
# print(pva)
# =============================================================================
