"""
Author: Alison Yao
Last Updated @ Jun 14, 2021
"""

import random
import numpy as np


def generate_random_N_paths(N, path_length):
    N_paths = []
    for _ in range(N):
        one_path = random.choices([0, 1], k=path_length)
        N_paths.append(one_path)
    return np.array(N_paths)

def check_feasibility(N_paths):
    '''
    s.t. constraints
    make sure initial paths & crossover paths & mutated paths are feasible
    '''
    link = np.sum(N_paths, axis=0)
    return link

def fitness():
    pass

def elitism():
    pass

def single_point_crossover():
    pass

def mutation():
    pass

def run_evolution():
    pass 


temp = check_feasibility(generate_random_N_paths(11, 3))
print(temp, type(temp))
