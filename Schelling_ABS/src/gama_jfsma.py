
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


global exp_id 
exp_id =[]

async def async_command_answer_handler(message: Dict):
 # print("Here is the answer to an async command: ", message)
  pass

async def gama_server_message_handler(message: Dict):
  # print("I just received a message from Gama-server and it's not an answer to a command!")
  # print("Here it is:", message)
   pass
async def main(n_cols, ppl_dens, soc_tol, map_size, perception_dist ):
    parameters = [
    {
      "type": "int",
   #   "value": "2",
      "value": str(n_cols),
      "name": "number_of_groups"
    },
    {
      "type": "float",
   #   "value": "0.7",
      "value": str(ppl_dens),
      "name": "density_of_people"
    },
    {
      "type": "float",
#      "value": "0.5",
      "value": str(soc_tol),
      "name": "percent_similar_wanted"
    },
    {
      "type": "int",
 #     "value": "40",
      "value": str(map_size),
      "name": "dimensions"
    },
    {
      "type": "int",
  #    "value": "2",
      "value": str(perception_dist),
      "name": "neighbours_distance"
  },
    {
      "type": "int",
  #    "value": "2",
      "value": str(0),
      "name": "num_sim"
    }
  ]
 #   for i in range(2) :
    #i = 0
    client1 = GamaSyncClient("localhost", 6868, async_command_answer_handler, gama_server_message_handler)
    client2 = GamaSyncClient("localhost", 6868, async_command_answer_handler, gama_server_message_handler)
    client3 = GamaSyncClient("localhost", 6868, async_command_answer_handler, gama_server_message_handler)
    client4 = GamaSyncClient("localhost", 6868, async_command_answer_handler, gama_server_message_handler)
    client5 = GamaSyncClient("localhost", 6868, async_command_answer_handler, gama_server_message_handler)
    
    await client1.connect(True)
    await client2.connect(True)
    await client3.connect(True)
    await client4.connect(True)
    await client5.connect(True)

    # Extract values to create filename
    param_values = [str(param["value"]) for param in parameters]    
    new_filename = "_".join(param_values) + "_exp"#+str(i)+".csv"
    output_path = r"C:\Users\psaves\Gama_Workspace2\Schelling_adapted_Paul\models\results\output"
    renamed_path = os.path.join(os.path.dirname(output_path), new_filename)
    f_pth = r"C:\Users\psaves\Gama_Workspace2\Schelling_adapted_Paul\models\Segregation (Agents).gaml"
   
    parameters[-1]["value"] = "1"       
    gama_response = client1.sync_load(f_pth, "schelling",None, None,None,None, parameters )          
    experiment_id = gama_response["content"]
    exp_id.append(experiment_id)     
    
    parameters[-1]["value"] = "2"      
    gama_response = client2.sync_load(f_pth, "schelling",None, None,None,None, parameters )          
    experiment_id = gama_response["content"]
    exp_id.append(experiment_id) 


    parameters[-1]["value"] = "3"      
    gama_response = client3.sync_load(f_pth, "schelling",None, None,None,None, parameters )          
    experiment_id = gama_response["content"]
    exp_id.append(experiment_id) 


    parameters[-1]["value"] = "4"      
    gama_response = client4.sync_load(f_pth, "schelling",None, None,None,None, parameters )          
    experiment_id = gama_response["content"]
    exp_id.append(experiment_id) 

    parameters[-1]["value"] = "5"      
    gama_response = client5.sync_load(f_pth, "schelling",None, None,None,None, parameters )          
    experiment_id = gama_response["content"]
    exp_id.append(experiment_id)     
        
    a = [True,True,True,True,True]
    c = -1
    while (np.max(a) and c<100) : 
        c= c+1
        print(c)
        if a[0] :
            if a[1] or a[2] or a[3] or a[4] : 
                await client1.step(exp_id[0],10,True)   
            else : 
                aa= client1.sync_step(exp_id[0],10,True)    
            with open(output_path + "1.csv", "r") as f1:
                last_line = f1.readlines()[-1]
            if (last_line[0]=="t") :
                a[0] = False
        if a[1] :
            if a[2] or a[3] or a[4] : 
                await client2.step(exp_id[1],10,True)
            else : 
                aa= client2.sync_step(exp_id[1],10,True)    
            with open(output_path + "2.csv", "r") as f2:
                last_line = f2.readlines()[-1]
            if (last_line[0]=="t") :
                a[1] = False
        if a[2] :    
            if a[3] or a[4] : 
                await client3.step(exp_id[2],10,True)
            else : 
                aa= client3.sync_step(exp_id[2],10,True)
                
            with open(output_path + "3.csv", "r") as f3:
                last_line = f3.readlines()[-1]
            if (last_line[0]=="t") :
                a[2] = False
        if a[3] :
            if a[4] : 
                await client4.step(exp_id[3],10,True)
            else : 
                aa= client4.sync_step(exp_id[3],10,True)
            with open(output_path + "4.csv", "r") as f4:
                last_line = f4.readlines()[-1]
            if (last_line[0]=="t") :
                a[3] = False
        if a[4] :
            aa= client5.sync_step(exp_id[4],10,True)
            with open(output_path + "5.csv", "r") as f5:
                last_line = f5.readlines()[-1]
            if (last_line[0]=="t") :
                a[4] = False
        await asyncio.sleep(1)

                  
    print(c,a)
    await client1.close_connection() 
    await client2.close_connection() 
    await client3.close_connection() 
    await client4.close_connection() 
    await client5.close_connection() 
    await asyncio.sleep(1)
    
    for i in range(5) :
      os.rename(output_path+str(i+1)+".csv", renamed_path+str(i)+".csv")
 
    
   
def run_simu(n_cols, ppl_dens, soc_tol, map_size, perception_dist ): 
        asyncio.run(main(n_cols, ppl_dens, soc_tol, map_size, perception_dist))
        

if __name__ == "__main__":
    
    design_space_theo = DesignSpace(
              [
                  IntegerVariable(2, 8),# Marche entre 1 et 8
                  FloatVariable(0.01, 0.99), #Density => Bornes discutables
                  FloatVariable(0.0, 1.0), # intolerance
                  IntegerVariable(10, 200), #size
                  IntegerVariable(1, 10), #vision
              ]
          )
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
    
    print(xt_200.shape)
    print(xt_100.shape)
    print(xt_50.shape)
    i= 171
    print(i)
    xt_200 = np.atleast_2d(xt_200[i])[0]
    xt_200[0] = 3
    xt_200[1] = 0.615
    xt_200[2] = 0.321
    xt_200[3] = 30
    xt_200[4] = 3
    xt_200 = np.atleast_2d(xt_200)
    
    
    for x in xt_200 :
        a = time.time()
        run_simu(int(x[0]),x[1],x[2],int(x[3]),int(x[4] ))
        print("time:", time.time()-a)
   
