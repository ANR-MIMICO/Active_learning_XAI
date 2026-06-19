
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:27:23 2024

@author: psaves
"""
import time
import asyncio
from gama_client.sync_client import GamaSyncClient
from typing import Dict

import numpy as np

import warnings
warnings.filterwarnings("ignore")




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
  
    await client1.connect(True)
    await client2.connect(True)
    
    param_values = [str(param["value"]) for param in parameters]    
    f_pth = r"C:\Users\psaves\Desktop\gama\gama.library\models\Toy Models\Segregation (Schelling)\models\Segregation (Agents).gaml"
   
    parameters[-1]["value"] = "1"       
    gama_response = client1.sync_load(f_pth, "schelling",None, None,None,None, parameters )          
    experiment_id = gama_response["content"]
    exp_id.append(experiment_id)     
    
    parameters[-1]["value"] = "2"      
    gama_response = client2.sync_load(f_pth, "schelling",None, None,None,None, parameters )          
    experiment_id = gama_response["content"]
    exp_id.append(experiment_id) 

    gama_response = client1.sync_expression(exp_id[0], r"sum_happy_people / number_of_people")
    print(gama_response)
    
    running = [True,True]
    c = -1
    while (c<5) : 
        c= c+1
        print(c)
        if running[0] :
            if running[1] : 
                await client1.step(exp_id[0],10,True)   
            else : 
                client1.sync_step(exp_id[0],10,True)    
            running[0] = False
        if running[1] :
            client2.sync_step(exp_id[1],10,True)    
            running[1] = False
    
        await asyncio.sleep(1)
    # sync step but in two differents threads
    gama_response = client1.sync_expression(exp_id[0], r"sum_happy_people / number_of_people")
    print(gama_response)    
  
    await client1.close_connection() 
    await client2.close_connection() 
    await asyncio.sleep(1)
  
   
def run_simu(n_cols, ppl_dens, soc_tol, map_size, perception_dist ): 
        asyncio.run(main(n_cols, ppl_dens, soc_tol, map_size, perception_dist))
        

if __name__ == "__main__":
    
  
    xt = np.zeros(6)
    xt[0] = 3
    xt[1] = 0.6
    xt[2] = 0.3
    xt[3] = 30
    xt[4] = 3
    xt= np.atleast_2d(xt)
    
    
    for x in xt :
        a = time.time()
        run_simu(int(x[0]),x[1],x[2],int(x[3]),int(x[4] ))
        print("time:", time.time()-a)
    
