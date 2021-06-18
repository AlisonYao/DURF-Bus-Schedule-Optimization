"""
Author: Alison Yao
Last Updated @ Jun 14, 2021
"""

import random
import numpy as np


def generate_random_N_paths(N, path_length):
    binary_N_paths = []
    for _ in range(N):
        # set the weights to initialize feasible solution faster
        one_path = random.choices(population=[0, 1], weights=[0.2, 0.8], k=path_length)
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
    objective function ish -> natural selection to pick the good ones
    the lower the better!!
    """
    total_cost = 0
    for one_path in binary_N_paths:
        target_indices = np.where(one_path == 1)[0]
        if len(target_indices) == 0:
            duration_interval_num = 0
        else:
            duration_interval_num = int(target_indices[-1] - target_indices[0] + 1)
        if duration_interval_num * intervalDuration <= 5:
            total_cost += 90
        elif duration_interval_num * intervalDuration <= 7.5:
            total_cost += 180
        else:
            total_cost += (20 * intervalDuration) * duration_interval_num
    return total_cost

def generate_population(population_size):
    population, fitness_scores = [], []
    while len(population) < population_size:
        binary_N_paths = generate_random_N_paths(N, intervalNum)
        if check_feasibility(binary_N_paths):
            population.append(binary_N_paths)
            fitness_score = fitness(binary_N_paths)
            fitness_scores.append(fitness_score)
            continue
        # else:
        #     print(".", end="")
    return np.array(population), np.array(fitness_scores)

def elitism(population, fitness_scores, elitism_cutoff=2):
    elite_indices = np.argpartition(np.array(fitness_scores), elitism_cutoff)[:elitism_cutoff]
    return population[elite_indices, :]

def crossover_mutation(population, fitness_scores, population_size, elitism_cutoff):
    """
    Randomly pick the good ones and cross them over
    """
    children = []
    while True:
        parents = random.choices(
            population=population,
            weights=[(max(fitness_scores) - score + 1)/(max(fitness_scores) * len(fitness_scores) - sum(fitness_scores) + len(fitness_scores)) for score in fitness_scores],
            k=2
        )
        kid1, kid2 = single_point_crossover(parents[0], parents[1])
        if kid1 is not None:
            kid1 = mutation(kid1)
            children.append(kid1)
        if len(children) == population_size - elitism_cutoff:
            return np.array(children)
        if kid2 is not None:
            kid2 = mutation(kid2)
            children.append(kid2)
        if len(children) == population_size - elitism_cutoff:
            return np.array(children)

def single_point_crossover(parent1, parent2):
    """
    Randomly pick the good ones and cross them over
    TODO: try multi-point
    TODO: try multi parents
    """
    assert parent1.size == parent2.size
    length = len(parent1)
    if length < 2:
        return parent1, parent2
    while True:
        cut = random.randint(1, length - 1)
        kid1 = np.append(parent1[0:cut, :], parent2[cut:, :]).reshape((N, intervalNum))
        kid2 = np.append(parent2[0:cut, :], parent1[cut:, :]).reshape((N, intervalNum))
        if check_feasibility(kid1) and check_feasibility(kid2):
            return kid1, kid2
        elif check_feasibility(kid1) and not check_feasibility(kid2):
            return kid1, None
        elif not check_feasibility(kid1) and check_feasibility(kid2):
            return None, kid2

def mutation(binary_N_paths):
    """
    Mutate only one node in one path for now
    TODO: try using a more complicated mutation method
    """
    while True:
        mutate_path = np.random.randint(0, N)
        mutate_node = np.random.randint(0, intervalNum)
        binary_N_paths[mutate_path][mutate_node] = abs(1 - binary_N_paths[mutate_path][mutate_node])
        if check_feasibility(binary_N_paths):
            return binary_N_paths

def result_stats(progress):
    """
    TODO: have a nicer output 
    TODO: TIME everything
    """
    print("Progress of improvement:", progress)

def run_evolution(population_size, evolution_depth, elitism_cutoff):
    # first initialize a population 
    population, population_fitnesses = generate_population(population_size)
    # keep track of improvement
    progress = []
    # start evolving :)
    for i in range(evolution_depth):
        progress.append(min(population_fitnesses))
        print(min(population_fitnesses))
        elites = elitism(population, population_fitnesses, elitism_cutoff)
        print('\nElites selected!')
        children = crossover_mutation(population, population_fitnesses, population_size, elitism_cutoff)
        print('\nChildren created!')
        population = np.concatenate([elites, children])
        print(f'----------------------------- generation {i + 1} evolved -----------------------------')
    result_stats(progress)

if __name__ == "__main__":
    # initialization for genetic algo
    population_size = 10
    evolution_depth = 30
    elitism_cutoff = 2
    # initialization
    N = 11 # # of buses
    D = 40 # #seats on each bus
    intervalDuration = 0.5
    # demand = np.array([
    #     [20, 10, 0], 
    #     [0, 50, 10]
    # ])
    demand = np.array([
        [114,106,132,132,117,83,57,52,13,8,18,13,26,3,13,10,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0], 
        [0,0,0,0,0,0,14,2,0,7,12,7,9,5,7,7,12,9,32,39,53,35,30,18,60,44,60,53,90,58,78,71,35,55]
    ])
    intervalNum = demand.shape[-1]

    # run main function
    run_evolution(population_size, evolution_depth, elitism_cutoff)
