# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:27:23 2024

@author: psaves
"""

import time
import asyncio
from gama_client.sync_client import GamaSyncClient
import numpy as np
from typing import Dict

global exp_id
exp_id = []
global clients
clients = []


global futures


async def async_command_answer_handler(message: Dict):
    global futures
    if "command" in message:
        futures[message["command"]["exp_id"]].set_result(message)


async def async_step_simulation(client, simulation_id, loop):
    global futures
    futures[simulation_id] = loop.create_future()
    await client.step(simulation_id, 10, True)
    res = await futures[simulation_id]


async def gama_server_message_handler(message: Dict):
    # print( message["content"])
    pass


def main(n_cols, ppl_dens, soc_tol, map_size, perception_dist):
    global futures
    futures = dict()
    parameters = [
        {
            "type": "int",
            #   "value": "2",
            "value": str(n_cols),
            "name": "number_of_groups",
        },
        {
            "type": "float",
            #   "value": "0.7",
            "value": str(ppl_dens),
            "name": "density_of_people",
        },
        {
            "type": "float",
            #      "value": "0.5",
            "value": str(soc_tol),
            "name": "percent_similar_wanted",
        },
        {
            "type": "int",
            #     "value": "40",
            "value": str(map_size),
            "name": "dimensions",
        },
        {
            "type": "int",
            #    "value": "2",
            "value": str(perception_dist),
            "name": "neighbours_distance",
        },
        {
            "type": "int",
            #    "value": "2",
            "value": str(0),
            "name": "num_sim",
        },
    ]
    #   for i in range(2) :
    # i = 0
    nb_clients = 10
    param_values = [str(param["value"]) for param in parameters]
    f_pth = r"C:\Users\psaves\Desktop\gama\gama.library\models\Toy Models\Segregation (Schelling)\models\Segregation (Agents).gaml"

    for i in range(0, nb_clients):
        clients.append(
            GamaSyncClient(
                "localhost",
                6868,
                async_command_answer_handler,
                gama_server_message_handler,
            )
        )
        clients[i].sync_connect(True)
        parameters[-1]["value"] = str(i + 1)
        gama_response = clients[i].sync_load(
            f_pth, "schelling", None, None, None, None, parameters
        )
        experiment_id = gama_response["content"]
        exp_id.append(experiment_id)

    loop = asyncio.get_event_loop()
    compt = -1
    happy = [False] * nb_clients
    groups = []
    while compt < 20 and not (np.min(happy)):
        compt += 1
        print(str(10 * compt) + " steps: ")

        for i in range(0, nb_clients):
            if not (happy[i]):
                gama_response = clients[i].sync_expression(
                    exp_id[i], r"sum_happy_people / number_of_people"
                )
                print(
                    "[sim" + str(i + 1) + "] hapiness is",
                    int(float(gama_response["content"]) * 100),
                    "%",
                )
                if float(gama_response["content"]) > 1 - 1e-10:
                    happy[i] = True
                    clients[i].sync_close_connection()
                else:
                    groups.append(
                        asyncio.gather(
                            async_step_simulation(clients[i], exp_id[i], loop),
                        )
                    )

        if not (np.min(happy)):
            all_groups = asyncio.gather(*groups)
            loop.run_until_complete(all_groups)

    for i in range(0, nb_clients):
        clients[i].sync_close_connection()
    try:
        loop.close()
    except RuntimeError:
        pass


def run_simu(n_cols, ppl_dens, soc_tol, map_size, perception_dist):
    main(n_cols, ppl_dens, soc_tol, map_size, perception_dist)


if __name__ == "__main__":
    xt = np.zeros(6)
    xt[0] = 3
    xt[1] = 0.61
    xt[2] = 0.61
    xt[3] = 35
    xt[4] = 3
    xt = np.atleast_2d(xt)

    for x in xt:
        a = time.time()
        run_simu(int(x[0]), x[1], x[2], int(x[3]), int(x[4]))
        print("time:", time.time() - a)
