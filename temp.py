import numpy as np


one_path = np.array([1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
num = 0
for i, node in enumerate(one_path):
    num += node
    if i+1 == len(one_path):
        print(num)
        print('Ends1')
        break
    if node == 1 and one_path[i+1] == 0:
        print(num)
        num = 0
print('Ends2')
