def single_point_crossover(parent1, parent2):
    """
    Randomly pick the good ones and cross them over
    """
    assert parent1.size == parent2.size
    length = len(parent1)
    if length < 2:
        return parent1, parent2
    cut = random.randint(1, length - 1)
    kid1 = np.append(parent1[0:cut, :], parent2[cut:, :]).reshape((N, intervalNum))
    kid2 = np.append(parent2[0:cut, :], parent1[cut:, :]).reshape((N, intervalNum))
    # print("c", end="")
    return kid1, kid2

def single_mutation(binary_N_paths):
    """
    Mutate only one node in one path for now
    """
    mutate_path = np.random.randint(0, N)
    mutate_node = np.random.randint(0, intervalNum)
    binary_N_paths[mutate_path][mutate_node] = abs(1 - binary_N_paths[mutate_path][mutate_node])
    # print("m", end="")
    return binary_N_paths
