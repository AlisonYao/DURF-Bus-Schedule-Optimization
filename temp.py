import numpy as np


parent1 = np.array([[[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]],
                    
                    [[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]],
                     
                    [[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]]])


parent2 = np.array([[[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]],
                     
                    [[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]]])

print(np.concatenate([parent1, parent2]))
