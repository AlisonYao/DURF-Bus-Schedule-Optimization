"""
Author: Alison Yao
Last Updated @ Jun 14, 2021
"""

import random
import numpy as np
from collections import defaultdict


def generate_random_N_paths(N, path_length):
    binary_N_paths = []
    for _ in range(N):
        one_path = random.choices([0, 1], k=path_length)
        binary_N_paths.append(one_path)
    return np.array(binary_N_paths)

def decode_one_path(one_path):
    decoded = []
    i, previous_node = None, None
    for j, current_node in enumerate(one_path):
        # first node
        if i == previous_node == None:
            if current_node == 0:
                decoded.append([1, 0, 0, 0])
            else:
                decoded.append([0, 1, 0, 0])
        # all nodes after first node
        else:
            previous_path = decoded[i]
            assert sum(previous_path) == 1
            if previous_path[0] == 1: # A
                if current_node == 0: # A
                    decoded.append([1, 0, 0, 0])
                else: # B
                    decoded.append([0, 1, 0, 0])
            elif previous_path[1] == 1: # B
                if current_node == 0: # D
                    decoded.append([0, 0, 0, 1])
                else: # C
                    decoded.append([0, 0, 1, 0])
            elif previous_path[2] == 1: # C
                if current_node == 0: # A
                    decoded.append([1, 0, 0, 0])
                else: # B
                    decoded.append([0, 1, 0, 0])
            else:
                if current_node == 0: # D
                    decoded.append([0, 0, 0, 1])
                else: # C
                    decoded.append([0, 0, 1, 0])
        i, previous_node = j, current_node
    return np.array(decoded).T

def check_feasibility(binary_N_paths):
    '''
    s.t. constraints
    make sure initial paths & crossover paths & mutated paths are feasible
    '''
    # get the link representation first
    directional_N_paths = [decode_one_path(one_path) for one_path in binary_N_paths]
    link = sum(directional_N_paths)
    # we hope every demand is met
    return np.greater_equal(link[1:3, :] * D, demand).all()

def fitness(binary_N_paths):
    """
    objective function ish
    """
    total_cost = 0
    for one_path in binary_N_paths:
        target_indices = np. where(one_path == 1)
        duration_interval_num = int(target_indices[0][-1] - target_indices[0][0] + 1)
        print(duration_interval_num)
        if duration_interval_num * intervalDuration <= 5:
            total_cost += 90
        elif duration_interval_num * intervalDuration <= 7.5:
            total_cost += 180
        else:
            total_cost += (20 * intervalDuration) * duration_interval_num
    return total_cost

def generate_population(population_size=20):
    population = defaultdict(int)
    for _ in range(population_size):
        while True:
            binary_N_paths = generate_random_N_paths(N, intervalNum)
            if check_feasibility(binary_N_paths):
                population[binary_N_paths] = fitness(binary_N_paths)
                break
            else:
                print("not feasible!!!!!!!!")
    print(population)

def elitism():
    pass

def single_point_crossover():
    pass

def mutation():
    pass

def run_evolution():
    pass 


if __name__ == "__main__":
    # initialization for genetic algo
    iteration = 10
    population_size = 10
    # initialization
    N = 3
    D = 40
    intervalNum = 3
    intervalDuration = 0.5
    demand = np.array([
        [50, 60, 20, ], 
        [0, 10, 50]
    ])
    # demand = np.array([
    #     [114,106,132,132,117,83,57,52,13,8,18,13,26,3,13,10,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0], 
    #     [0,0,0,0,0,0,14,2,0,7,12,7,9,5,7,7,12,9,32,39,53,35,30,18,60,44,60,53,90,58,78,71,35,55]
    # ])
    
    # temp = generate_random_N_paths(N, intervalNum)
    # print(fitness(temp))
    print(generate_population())
