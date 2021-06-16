import numpy as np
def mutation(binary_N_paths):
    mutate_path = np.random.randint(0, N)
    mutate_node = np.random.randint(0, intervalNum)
    print(binary_N_paths)
    binary_N_paths[mutate_path][mutate_node] = abs(1 - binary_N_paths[mutate_path][mutate_node])
    print(mutate_path, mutate_node)
    print(binary_N_paths)


binary_N_paths = np.array([[0, 1, 1],
                            [1, 1, 0],
                            [1, 1, 1]])
N = 3
intervalNum = 3
mutation(binary_N_paths)
