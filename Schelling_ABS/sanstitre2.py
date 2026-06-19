# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:30:49 2025

@author: psaves
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import RBF

# Initialize with optional custom kernel
kernel = 1.0 * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)


# export_gpr_to_pmml.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml

# 1. Generate sample data
np.random.seed(0)
X = np.random.rand(50, 1)
y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * np.random.randn(50)

# 2. Configure GPR with RBF kernel
kernel = 1.0 * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

# 3. Wrap in a PMMLPipeline
pipeline = PMMLPipeline([
    ("gpr", GaussianProcessRegressor())   # Final estimator must be a regressor
])

# 4. Fit the pipeline
pipeline.fit(X, y)

# 5. Export to PMML
#    JVM_OPTIONS can be set e.g. in environment or via Java system properties
sklearn2pmml(
    pipeline,
    "gpr_model.pmml",
    with_repr=True
)
print("PMML file 'gpr_model.pmml' created.")
