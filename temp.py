import numpy as np

def demand_constraint(binary_N_paths):
    link = np.array([
        [2, 2, 2],
        [3, 3, 3],
        [4, 1, 4],
        [5, 5, 5]
    ])
    x = np.greater_equal(demand, link[1:3, :])
    y = demand - link[1:3, :]
    print(x)
    print(y)
    print(x * y)
    print(np.sum(x * y))


D = 40
demand = np.array([
    [2, 2, 2],
    [3, 3, 3]
])

demand_constraint(demand)
