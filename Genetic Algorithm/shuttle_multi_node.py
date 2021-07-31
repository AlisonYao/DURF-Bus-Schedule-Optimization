"""
Author: Alison Yao (yy2564@nyu.edu)
Last Updated @ July 31, 2021
"""

import random
import numpy as np
import time
import matplotlib.pyplot as plt


def generate_random_N_paths(N, path_length):
    '''
    Randomize N paths where 1 path is like 00 01 00 01 01 01
    '''
    one_solution = []
    while len(one_solution) < N:
        one_path_single_digit = random.choices(population=[0, 1], weights=[1-initial_prob, initial_prob], k=path_length)
        one_path_double_digit = ''
        for i in one_path_single_digit:
            if i == 0:
                one_path_double_digit += '00'
            elif i == 1:
                one_path_double_digit += random.choices(population=['10', '01'], weights=[1-jinyang_prob, jinyang_prob])[0]
        if check_path_integrity(one_path_double_digit):
            one_solution.append(one_path_double_digit)
    return one_solution

def check_path_integrity(one_path_double_digit):
    last_visited = None
    for i in range(len(one_path_double_digit)):
        if i % 2 == 0:
            two_digits = one_path_double_digit[i:i+2]
            if two_digits != '00':
                # first time going to AB
                if last_visited is None:
                    last_visited = 'AB'
                # following times
                elif last_visited == 'GC':
                    if two_digits == '01':
                        return False
                    else: # '10'
                        last_visited = 'AB'
                elif last_visited == 'PS':
                    if two_digits == '10':
                        return False
                    else: # '01'
                        last_visited = 'AB'
                else:
                    if two_digits == '10':
                        last_visited = 'GC'
                    else: # '01'
                        last_visited = 'PS'
    return True

def decode_one_path(one_path_double_digit):
    decoded, initial_node, last_visited = [], None, None
    for i in range(len(one_path_double_digit)):
        if i % 2 == 0:
            two_digits = one_path_double_digit[i:i+2]
            if two_digits == '00':
                if last_visited is None:
                    decoded.append([0, 0, 0, 0, 0, 0, 0])
                elif last_visited == 'GC':
                    decoded.append([1, 0, 0, 0, 0, 0, 0])
                elif last_visited == 'AB':
                    decoded.append([0, 0, 0, 1, 0, 0, 0])
                else: # PS
                    decoded.append([0, 0, 0, 0, 0, 0, 1])
            elif two_digits == '10':
                if last_visited is None:
                    initial_node = 0
                    last_visited = 'AB'
                    decoded.append([0, 1, 0, 0, 0, 0, 0])
                elif last_visited == 'AB':
                    last_visited = 'GC'
                    decoded.append([0, 0, 1, 0, 0, 0, 0])
                elif last_visited == 'GC':
                    last_visited = 'AB'
                    decoded.append([0, 1, 0, 0, 0, 0, 0])
                else:
                    print('SOMETHING IS WRONG1!!!')
            elif two_digits == '01':
                if last_visited is None:
                    initial_node = -1
                    last_visited = 'AB'
                    decoded.append([0, 0, 0, 0, 0, 1, 0])
                elif last_visited == 'AB':
                    last_visited = 'PS'
                    decoded.append([0, 0, 0, 0, 1, 0, 0])
                elif last_visited == 'PS':
                    last_visited = 'AB'
                    decoded.append([0, 0, 0, 0, 0, 1, 0])
                else:
                    print('SOMETHING IS WRONG2!!!')
            print(two_digits, initial_node, last_visited)
    decoded = np.array(decoded).T
    decoded_sum = decoded.sum(axis=0)
    k = 0
    while decoded_sum[k] == 0:
        decoded[initial_node, k] = 1
        k += 1
    return decoded

def meet_demand(binary_N_paths, tolerance):
    '''
    meet demand
    '''
    pass

def rush_hour_constraint(binary_N_paths):
    '''
    during rush hours, one interval is not enough time to commute
    '''
    pass

def max_working_hour_constraint(binary_N_paths):
    '''
    make sure that no driver works more than a few hours continuously
    '''
    pass

def check_feasibility(binary_N_paths, checkRushHour=False, checkMaxWorkingHour=False):
    '''
    s.t. constraints (make sure initial paths & crossover paths & mutated paths are feasible)
    constraint1: meet demand
    constraint2: during rush hours, one interval is not enough time to commute (optional)
    constraint3: make sure that no driver works more than a few hours continuously (optional)
    '''
    pass

def fitness(binary_N_paths, addPenalty=False):
    """
    objective function ish -> natural selection to pick the good ones
    the lower the better!!
    """
    pass

def generate_population(population_size):
    pass

def elitism(population, fitness_scores, elitism_cutoff=2):
    pass

def crossover_mutation(population, fitness_scores, population_size, elitism_cutoff):
    """
    Randomly pick the good ones and cross them over
    """
    pass

def single_point_crossover(parent1, parent2):
    """
    Randomly pick the good ones and cross them over
    """
    pass

def single_mutation(binary_N_paths):
    """
    Mutate only one node in one path for now
    """
    pass  

def result_stats(progress_with_penalty, progress):
    """
    print important stats & visulize progress_with_penalty
    """
    pass

def run_evolution(population_size, evolution_depth, elitism_cutoff):
    '''
    Main function of Genetic Algorithm
    '''
    pass


if __name__ == "__main__":

    """initialization for genetic algo"""
    initial_prob = 0.8
    jinyang_prob = 0.1
    print(generate_random_N_paths(3, 4))
    # population_size = 20
#     elitism_cutoff = 2
#     mutation_num = 1
#     loop_limit = 100
#     evolution_depth = 1000

#     """initialization for buses"""
#     # # of buses
#     N = 11
#     # #seats on each bus
#     D = 40
#     tolerance = 0
#     intervalDuration = 0.5
    # numerical example 1
    demand_GC = np.array([
        [114,106,132,132,117,83,57,52,13,8,18,13,26,3,13,10,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0], 
        [0,0,0,0,0,0,14,2,0,7,12,7,9,5,7,7,12,9,32,39,53,35,30,18,60,44,60,53,90,58,78,71,35,55]
    ])
    demand_PS = demand_GC / 9
    demand_PS.astype(int)
#     # numerical example 2
#     demand = demand * 0.5
#     demand.astype(int)
#     # toy numerical example
#     # demand = np.array([
#     #     [60, 120, 60,  10,  0,  0,  0], 
#     #     [ 0,  0, 40, 60, 100, 20, 20]
#     # ])

#     intervalNum = demand.shape[-1]
#     maxWorkingHour = 4
#     checkRushHourFlag = True
#     checkMaxWorkingHourFlag = True
#     rushHourViolationPenalty = 7
#     maxWorkingHourViolationPenalty = 5

#     # run main function
#     run_evolution(population_size, evolution_depth, elitism_cutoff)
