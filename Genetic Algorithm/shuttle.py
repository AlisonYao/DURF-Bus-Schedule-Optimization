"""
Author: Alison Yao (yy2564@nyu.edu)
Last Updated @ July 28, 2021
"""

import random
import numpy as np
import time
import matplotlib.pyplot as plt


def generate_random_N_paths(N, path_length):
    binary_N_paths = []
    for _ in range(N):
        # set the weights to initialize feasible solution faster
        one_path = random.choices(population=[0, 1], weights=[1-initial_prob, initial_prob], k=path_length)
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

def meet_demand(binary_N_paths, tolerance):
    '''
    meet demand
    '''
    # get the link representation first
    directional_N_paths = [decode_one_path(one_path) for one_path in binary_N_paths]
    link = sum(directional_N_paths)
    # we hope every demand is met
    return np.greater_equal(link[1:3, :] * D, demand - tolerance).all()

def rush_hour_constraint(binary_N_paths):
    '''
    during rush hours, one interval is not enough time to commute
    '''
    violationCount = 0
    for one_path in binary_N_paths:
        # morning rush hour
        if one_path[1] + one_path[2] == 2 or one_path[2] + one_path[3] == 2:
            violationCount += 1
        # evening rush hour
        if one_path[21] + one_path[22] == 2:
            violationCount += 1
        # if one_path[4] + one_path[5] == 2:
        #     violationCount += 1
    return int(violationCount) == 0, int(violationCount)

def max_working_hour_constraint(binary_N_paths):
    '''
    make sure that no driver works more than a few hours continuously
    '''
    violationCount = 0
    for one_path in binary_N_paths:
        num, num_list = 0, []
        for i, node in enumerate(one_path):
            num += node
            if i+1 == len(one_path):
                num_list.append(num)
                continue
            if node == 1 and one_path[i+1] == 0:
                num_list.append(num)
                num = 0
        violationCount += sum(np.array(num_list) > maxWorkingHour / intervalDuration)
    return violationCount == 0, violationCount

def check_feasibility(binary_N_paths, checkRushHour=False, checkMaxWorkingHour=False):
    '''
    s.t. constraints (make sure initial paths & crossover paths & mutated paths are feasible)
    constraint1: meet demand
    constraint2: during rush hours, one interval is not enough time to commute (optional)
    constraint3: make sure that no driver works more than a few hours continuously (optional)
    '''
    # print('binary_N_paths:\n', binary_N_paths)
    rushHour, maxWorkingHour = True, True
    if checkRushHour:
        rushHour, rushHourViolationNum = rush_hour_constraint(binary_N_paths)
    if checkMaxWorkingHour:
        maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(binary_N_paths)
    demandFlag = meet_demand(binary_N_paths, tolerance)
    # if not demandFlag:
    #     print("d", end="")
    if not rushHour:
        print("r"+str(rushHourViolationNum), end="")
    if not maxWorkingHour:
        print("w"+str(maxWorkingHourViolationNum), end="")
    return demandFlag and rushHour and maxWorkingHour

def fitness(binary_N_paths, addPenalty=False):
    """
    objective function ish -> natural selection to pick the good ones
    the lower the better!!
    """
    total_cost = 0
    # basic cost
    for one_path in binary_N_paths:
        target_indices = np.where(one_path == 1)[0]
        if len(target_indices) == 0:
            duration_interval_num = 0
        else:
            duration_interval_num = int(target_indices[-1] - target_indices[0] + 1)
        if duration_interval_num == 0:
            total_cost += 0
        elif duration_interval_num * intervalDuration <= 5:
            total_cost += 90
        elif duration_interval_num * intervalDuration <= 7.5:
            total_cost += 180
        else:
            total_cost += (20 * intervalDuration) * duration_interval_num
    # add penalty
    if addPenalty:
        rushHour, rushHourViolatonNum = rush_hour_constraint(binary_N_paths)
        maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(binary_N_paths)
        total_cost += rushHourViolatonNum * rushHourViolationPenalty + maxWorkingHourViolationNum * maxWorkingHourViolationPenalty
    return total_cost

def generate_population(population_size):
    population, fitness_scores_add_penalty = [], []
    while len(population) < population_size:
        binary_N_paths = generate_random_N_paths(N, intervalNum)
        if check_feasibility(binary_N_paths):
            population.append(binary_N_paths)
            fitness_score_add_penalty = fitness(binary_N_paths, addPenalty=True)
            fitness_scores_add_penalty.append(fitness_score_add_penalty)
            continue
        # else:
        #     print("i", end="")
    return np.array(population), np.array(fitness_scores_add_penalty)

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
            for _ in range(mutation_num):
                kid1 = single_mutation(kid1)
            children.append(kid1)
        if len(children) == population_size - elitism_cutoff:
            return np.array(children)
        if kid2 is not None:
            for _ in range(mutation_num):
                kid2 = single_mutation(kid2)
            children.append(kid2)
        if len(children) == population_size - elitism_cutoff:
            return np.array(children)

def single_point_crossover(parent1, parent2):
    """
    Randomly pick the good ones and cross them over
    """
    assert parent1.size == parent2.size
    length = len(parent1)
    if length < 2:
        return parent1, parent2
    count = 0
    while count <= loop_limit:
        cut = random.randint(1, length - 1)
        kid1 = np.append(parent1[0:cut, :], parent2[cut:, :]).reshape((N, intervalNum))
        kid2 = np.append(parent2[0:cut, :], parent1[cut:, :]).reshape((N, intervalNum))
        if check_feasibility(kid1) and check_feasibility(kid2):
            return kid1, kid2
        elif check_feasibility(kid1) and not check_feasibility(kid2):
            return kid1, None
        elif not check_feasibility(kid1) and check_feasibility(kid2):
            return None, kid2
        # else:
            # print("c", end="")
        count += 1
    return parent1, parent2

def single_mutation(binary_N_paths):
    """
    Mutate only one node in one path for now
    """
    count = 0
    binary_N_paths_copy = binary_N_paths.copy()
    while count <= loop_limit:
        mutate_path = np.random.randint(0, N)
        mutate_node = np.random.randint(0, intervalNum)
        binary_N_paths_copy[mutate_path][mutate_node] = abs(1 - binary_N_paths_copy[mutate_path][mutate_node])
        if check_feasibility(binary_N_paths_copy):
            return binary_N_paths_copy
        # else:
        #     print("m", end="")
        count += 1
    return binary_N_paths    

def result_stats(progress_with_penalty, progress):
    """
    print important stats & visulize progress_with_penalty
    """
    print('**************************************************************')
    print("Progress_with_penalty of improvement:", progress_with_penalty)
    print("Progress of improvement:", progress)
    print("Improvement Rate of progress:", abs(progress[-1] - progress[0])/progress[0])
    print('**************************************************************')
    plt.plot(progress_with_penalty, data=progress_with_penalty, label='with penalty')
    plt.plot(progress, data=progress, label='no penalty')
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

def run_evolution(population_size, evolution_depth, elitism_cutoff):
    '''
    Main function of Genetic Algorithm
    '''
    tic = time.time()

    # first initialize a population 
    population, population_fitnesses_add_penalty = generate_population(population_size)
    initialization_end = time.time()
    print('\nInitialization Done!', initialization_end - tic)
    population_fitnesses = [fitness(binary_N_paths) for binary_N_paths in population]
    print(f'Initial Min Cost: {min(population_fitnesses_add_penalty)} -> {min(population_fitnesses)}')
    # keep track of improvement
    progress_with_penalty, progress = [], []
    
    # start evolving :)
    for i in range(evolution_depth):
        progress_with_penalty.append(min(population_fitnesses_add_penalty))
        progress.append(min(population_fitnesses))
        print(f'----------------------------- generation {i + 1} Start! -----------------------------')
        elitism_begin = time.time()
        elites = elitism(population, population_fitnesses_add_penalty, elitism_cutoff)
        elitism_end = time.time()
        print('Elites selected!')
        children_begin = time.time()
        children = crossover_mutation(population, population_fitnesses_add_penalty, population_size, elitism_cutoff)
        children_end = time.time()
        print('Children created!')
        population = np.concatenate([elites, children])
        population_fitnesses_add_penalty = [fitness(binary_N_paths, addPenalty=True) for binary_N_paths in population]
        population_fitnesses = [fitness(binary_N_paths) for binary_N_paths in population]
        
        evol_end = time.time()
        print(f"Min Cost: {min(population_fitnesses_add_penalty)} -> {min(population_fitnesses)}")
        # check best solution feasibility
        minIndex = population_fitnesses_add_penalty.index(min(population_fitnesses_add_penalty))
        best_solution = population[minIndex]
        allFeasibilityFlag = check_feasibility(best_solution, checkRushHour=checkRushHourFlag, checkMaxWorkingHour=checkMaxWorkingHourFlag)
        print("\nAll constraints met?", allFeasibilityFlag)

        # print best solution
        print('best solution (path):\n', best_solution)
        directional_N_paths = [decode_one_path(one_path) for one_path in population[minIndex]]
        link = sum(directional_N_paths)
        print('best solution (link): \n', link)

        print(f'---------------------- generation {i + 1} evolved! Time: {evol_end - elitism_begin:.4f}s ----------------------\n')
    
    # plot results
    result_stats(progress_with_penalty, progress)

    # print best solution
    minIndex = population_fitnesses_add_penalty.index(min(population_fitnesses_add_penalty))
    best_solution = population[minIndex]
    print('best solution (path):\n', best_solution)

    # check if all constraints are met (ideally True)
    print("\nAll constraints met?", check_feasibility(best_solution, checkRushHour=checkRushHourFlag, checkMaxWorkingHour=checkMaxWorkingHourFlag))
    directional_N_paths = [decode_one_path(one_path) for one_path in population[minIndex]]
    link = sum(directional_N_paths)
    print('best solution (link): \n', link)


if __name__ == "__main__":

    """initialization for genetic algo"""
    initial_prob = 0.8
    population_size = 20
    elitism_cutoff = 2
    mutation_num = 1
    loop_limit = 100
    evolution_depth = 1000

    """initialization for buses"""
    # # of buses
    N = 11
    # #seats on each bus
    D = 40
    tolerance = 0
    intervalDuration = 0.5
    # numerical example 1
    demand = np.array([
        [114,106,132,132,117,83,57,52,13,8,18,13,26,3,13,10,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0], 
        [0,0,0,0,0,0,14,2,0,7,12,7,9,5,7,7,12,9,32,39,53,35,30,18,60,44,60,53,90,58,78,71,35,55]
    ])
    # numerical example 2
    demand = demand * 0.5
    demand.astype(int)
    # toy numerical example
    # demand = np.array([
    #     [60, 120, 60,  10,  0,  0,  0], 
    #     [ 0,  0, 40, 60, 100, 20, 20]
    # ])

    intervalNum = demand.shape[-1]
    maxWorkingHour = 4
    checkRushHourFlag = True
    checkMaxWorkingHourFlag = True
    rushHourViolationPenalty = 7
    maxWorkingHourViolationPenalty = 5

    # run main function
    run_evolution(population_size, evolution_depth, elitism_cutoff)
