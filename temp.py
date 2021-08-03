import numpy as np
import random

def single_mutation(binary_N_paths):
    """
    Mutate only one node in one path for now
    """
    mutate_path = np.random.randint(0, N)
    mutate_index = np.random.randint(0, intervalNum) * 2
    double_digits_to_mutate = binary_N_paths[mutate_path][mutate_index:mutate_index+2]
    pool = ['00', '01', '10']
    pool.remove(double_digits_to_mutate)
    mutated_double_digits = random.choices(population=pool)[0]
    original_string = binary_N_paths[mutate_path]
    binary_N_paths[mutate_path] = original_string[:mutate_index] + mutated_double_digits + original_string[mutate_index+2:]

N = 2
intervalNum = 8
binary_N_paths = ['1010101000100001', '0010101000100001']
print(single_mutation(binary_N_paths))


