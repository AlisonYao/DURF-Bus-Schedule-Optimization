"""
Author: Alison Yao (yy2564@nyu.edu)
Last Updated @ August 14, 2021

version 2 converts the demand into penalty
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
                one_path_double_digit += random.choices(population=['10', '01'], weights=[1-pusan_prob, pusan_prob])[0]
        if check_path_integrity(one_path_double_digit):
            one_solution.append(one_path_double_digit)
    return one_solution

def check_solution_integrity(solution):
    for one_path_double_digit in solution:
        if not check_path_integrity(one_path_double_digit):
            return False
    return True

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
                elif last_visited == 'JQJY':
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
                        last_visited = 'JQJY'
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
                elif last_visited == 'JQJY':
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
                    last_visited = 'JQJY'
                    decoded.append([0, 0, 1, 0, 0, 0, 0])
                elif last_visited == 'JQJY':
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
    decoded = np.array(decoded).T
    decoded_sum = decoded.sum(axis=0)
    if sum(decoded_sum) == 0:
        if random.random() <= pusan_prob:
            decoded[0, :] = 0
        else:
            decoded[0, :] = 1
        return decoded
    k = 0
    while decoded_sum[k] == 0:
        decoded[initial_node, k] = 1
        k += 1
    return decoded

def demand_constraint(binary_N_paths, tolerance):
    '''
    make sure the demand is met
    '''
    directional_N_paths = [decode_one_path(one_path) for one_path in binary_N_paths]
    link = sum(directional_N_paths)
    link_JQJY = link[:4, :]
    link_PS = link[-1:2:-1, :]
    JQJY_supply_demand_difference = np.greater_equal(demand_JQJY - tolerance, link_JQJY[1:3, :] * D)
    JQJY_mask = (demand_JQJY - tolerance) - (link_JQJY[1:3, :] * D)
    PS_supply_demand_difference = np.greater_equal(demand_PS - tolerance, link_PS[1:3, :] * D)
    PS_mask = (demand_PS - tolerance) - (link_PS[1:3, :] * D)
    missedDemandNumJQJY = np.sum(JQJY_supply_demand_difference * JQJY_mask)
    missedDemandNumPS = np.sum(PS_supply_demand_difference * PS_mask)
    return int(missedDemandNumJQJY + missedDemandNumPS) == 0, int(missedDemandNumJQJY + missedDemandNumPS)

def rush_hour_constraint(binary_N_paths):
    '''
    during rush hours, one interval is not enough time to commute
    '''
    violationCount = 0
    for one_path_double_digit in binary_N_paths:
        one_path_single_digit_list = []
        one_path_double_digit_list = list(one_path_double_digit)
        for i in range(len(one_path_double_digit_list)):
            if i % 2 == 0:
                one_path_single_digit_list.append(int(one_path_double_digit_list[i]) + int(one_path_double_digit_list[i+1]))
        # morning rush hour
        if one_path_single_digit_list[1] + one_path_single_digit_list[2] == 2:
            violationCount += 1
        # evening rush hour
        if one_path_single_digit_list[21] + one_path_single_digit_list[22] == 2:
            violationCount += 1
    return int(violationCount) == 0, int(violationCount)

def max_working_hour_constraint(binary_N_paths):
    '''
    make sure that no driver works more than a few hours continuously
    '''
    violationCount = 0
    for one_path_double_digit in binary_N_paths:
        one_path_single_digit_list = []
        one_path_double_digit_list = list(one_path_double_digit)
        for i in range(len(one_path_double_digit_list)):
            if i % 2 == 0:
                one_path_single_digit_list.append(int(one_path_double_digit_list[i]) + int(one_path_double_digit_list[i+1]))
        num, num_list = 0, []
        one_path_copy = one_path_single_digit_list.copy()
        # first check if rush hour 10 actually is 11.
        if checkRushHourFlag:
            if one_path_copy[1] == 1 and one_path_copy[2] == 0:
                one_path_copy[2] = 1
            if one_path_copy[21] == 1 and one_path_copy[22] == 0:
                one_path_copy[22] = 1
        for i, node in enumerate(one_path_copy):
            num += node
            if i+1 == len(one_path_copy):
                num_list.append(num)
                continue
            if node == 1 and one_path_copy[i+1] == 0:
                num_list.append(num)
                num = 0
        violationCount += sum(np.array(num_list) > maxWorkingHour / intervalDuration)
    return int(violationCount) == 0, int(violationCount)

def check_feasibility(binary_N_paths, checkDemand=True, checkRushHour=False, checkMaxWorkingHour=False):
    '''
    s.t. constraints (make sure initial paths & crossover paths & mutated paths are feasible)
    constraint1: meet demand
    constraint2: during rush hours, one interval is not enough time to commute (optional)
    constraint3: make sure that no driver works more than a few hours continuously
    '''
    demandFlag, rushHour, maxWorkingHour = True, True, True
    if checkDemand:
        demandFlag, demandViolationNum = demand_constraint(binary_N_paths, tolerance)
    if checkRushHour:
        rushHour, rushHourViolationNum = rush_hour_constraint(binary_N_paths)
    if checkMaxWorkingHour:
        maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(binary_N_paths)
    if not demandFlag:
        print("d"+str(demandViolationNum), end="")
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
    for one_path_double_digit in binary_N_paths:
        one_path_single_digit_list = []
        one_path_double_digit_list = list(one_path_double_digit)
        for i in range(len(one_path_double_digit_list)):
            if i % 2 == 0:
                one_path_single_digit_list.append(int(one_path_double_digit_list[i]) + int(one_path_double_digit_list[i+1]))
        one_path_single_digit_np = np.array(one_path_single_digit_list)
        target_indices = np.where(one_path_single_digit_np == 1)[0]
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
        demandFlag, demandViolationNum = demand_constraint(binary_N_paths, tolerance)
        rushHour, rushHourViolatonNum = rush_hour_constraint(binary_N_paths)
        maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(binary_N_paths)
        if checkDemandFlag:
            total_cost += alpha * demandViolationNum * demandViolationPenalty
        if checkRushHourFlag:
            total_cost += rushHourViolatonNum * rushHourViolationPenalty
        if maxWorkingHourViolationPenalty:
            total_cost += maxWorkingHourViolationNum * maxWorkingHourViolationPenalty
    return total_cost

def generate_population(population_size):
    population, fitness_scores_add_penalty = [], []
    for _ in range(population_size):
        binary_N_paths = generate_random_N_paths(N, intervalNum)
        population.append(binary_N_paths)
        fitness_score_add_penalty = fitness(binary_N_paths, addPenalty=True)
        fitness_scores_add_penalty.append(fitness_score_add_penalty)
    return np.array(population), np.array(fitness_scores_add_penalty)

def elitism(population, fitness_scores, elitism_cutoff=2):
    elite_indices = np.argpartition(np.array(fitness_scores), elitism_cutoff)[:elitism_cutoff]
    return population[elite_indices, :]

def create_next_generation(population, population_fitnesses_add_penalty, population_size, elitism_cutoff):
    """
    Randomly pick the good ones and cross them over
    """
    children = []
    while True:
        parents = random.choices(
            population=population,
            weights=[(max(population_fitnesses_add_penalty) - score + 1)/(max(population_fitnesses_add_penalty) * len(population_fitnesses_add_penalty) - sum(population_fitnesses_add_penalty) + len(population_fitnesses_add_penalty)) for score in population_fitnesses_add_penalty],
            k=2
        )
        kid1, kid2 = single_point_crossover(parents[0], parents[1])
        for _ in range(mutation_num):
            kid1 = single_mutation(kid1)
        children.append(kid1)
        if len(children) == population_size - elitism_cutoff:
            return np.array(children)
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
        cut = random.randint(1, length - 1) * 2
        kid1 = np.array(list(parent1)[:cut] + list(parent2)[cut:])
        kid2 = np.array(list(parent2)[:cut] + list(parent1)[cut:])
        if check_solution_integrity(kid1) and check_solution_integrity(kid2):
            return kid1, kid2
        elif check_solution_integrity(kid1) and not check_solution_integrity(kid2):
            return kid1, None
        elif not check_solution_integrity(kid1) and check_solution_integrity(kid2):
            return None, kid2
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
        mutate_index = np.random.randint(0, intervalNum) * 2
        double_digits_to_mutate = binary_N_paths_copy[mutate_path][mutate_index:mutate_index+2]
        pool = ['00', '01', '10']
        pool.remove(double_digits_to_mutate)
        mutated_double_digits = random.choices(population=pool)[0]
        original_string = binary_N_paths_copy[mutate_path]
        mutated_string = original_string[:mutate_index] + mutated_double_digits + original_string[mutate_index+2:]
        if check_path_integrity(mutated_string):
            binary_N_paths_copy[mutate_path] = mutated_string
            return binary_N_paths_copy
        count += 1
    return binary_N_paths

def result_stats(progress_with_penalty, progress):
    """
    print important stats & visulize progress_with_penalty
    """
    print('**************************************************************')
    print(f"Progress_with_penalty of improvement: {progress_with_penalty[0]} to {progress_with_penalty[-1]}" )
    print(f"Progress of improvement: {progress[0]} to {progress[-1]}")
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
    print(f'\nInitialization Done! Time: {initialization_end - tic:.6f}s')
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
        print('Elites selected!')
        children = create_next_generation(population, population_fitnesses_add_penalty, population_size, elitism_cutoff)
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

if __name__ == "__main__":

    """initialization for genetic algo"""
    initial_prob = 0.8
    pusan_prob = 0.2
    population_size = 20
    elitism_cutoff = 2
    mutation_num = 1
    loop_limit = 100
    evolution_depth = 50

    """initialization for buses"""
    # # of buses
    N = 10
    # #seats on each bus
    D = 50
    tolerance = 0
    intervalDuration = 0.5
    # numerical example
    demand = np.array([
        [114,106,132,132,117,83,57,52,13,8,18,13,26,3,13,10,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0], 
        [0,0,0,0,0,0,14,2,0,7,12,7,9,5,7,7,12,9,32,39,53,35,30,18,60,44,60,53,90,58,78,71,35,55]
    ])
    demand_JQJY = np.around(demand / 10)
    demand_JQJY = demand_JQJY.astype(int)
    demand_PS = np.around(demand / 90)
    demand_PS = demand_PS.astype(int)

    print(demand_JQJY)
    print(demand_PS)

    intervalNum = demand.shape[-1]
    maxWorkingHour = 4
    checkDemandFlag, checkRushHourFlag, checkMaxWorkingHourFlag = True, True, True
    alpha, demandViolationPenalty, rushHourViolationPenalty, maxWorkingHourViolationPenalty = 1, 10, 7, 5

    # run main function
    run_evolution(population_size, evolution_depth, elitism_cutoff)
