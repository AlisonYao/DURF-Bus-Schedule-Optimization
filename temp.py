import numpy as np

directional_N_paths = np.array([
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1]
])
directional_N_paths[0, :] = 3
print(directional_N_paths)
