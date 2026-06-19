# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 10:19:45 2025

@author: psaves
"""

import argparse
import numpy as np
import pandas as pd
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import openturns.viewer as otv

parser = argparse.ArgumentParser()
parser.add_argument("--target", choices=["price", "symbiose"], default="price")
args = parser.parse_args()

problem = {
    'num_vars': 5,
    'names': ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost'],
    'bounds': [
        [0, 200],
        [0.1, 2],
        [1e-5, 1e-2],
        [0, 1],
        [0.1, 1]
    ]
}


df_results = pd.read_csv("sobol_batch_raw_results.csv")
df_mean = df_results.groupby(['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']).mean().reset_index()

Y = df_mean[args.target].values
#Y = df_results[args.target].values
#Y_standardized = (Y - Y.mean()) / np.sqrt(Y.var())
#Si = sobol.analyze(problem, Y_standardized, calc_second_order=True, print_to_console=True)

import openturns as ot
import pandas as pd

# Assuming df_results is your DataFrame
input_names = ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']
target_name = 'symbiose'

# Convert inputs and outputs to OpenTURNS Sample
X_sample = ot.Sample(df_mean[input_names].values.tolist())
Y_sample = ot.Sample(df_mean[[target_name]].values.tolist())


chaosAlgo = ot.FunctionalChaosAlgorithm(X_sample, Y_sample) 

chaosAlgo.run()
chaosResult = chaosAlgo.getResult()
metamodel = chaosResult.getMetaModel()
chaosSI = ot.FunctionalChaosSobolIndices(chaosResult)
first_order = [chaosSI.getSobolIndex(i) for i in range(5)]


total_order = [chaosSI.getSobolTotalIndex(i) for i in range(5)]
graph = ot.SobolIndicesAlgorithm.DrawSobolIndices(input_names, first_order, total_order)
view = otv.View(graph)


