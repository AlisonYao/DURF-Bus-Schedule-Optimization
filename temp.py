import numpy as np


one_path = np.array([1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
# num = 0
# for i, node in enumerate(one_path):
#     num += node
#     if i+1 == len(one_path):
#         print(num)
#         print('Ends1')
#         break
#     if node == 1 and one_path[i+1] == 0:
#         print(num)
#         num = 0


# for one_path in binary_N_paths:
#     num = 0
#     for i, node in enumerate(one_path):
#         num += node
#         if i+1 == len(one_path):
#             if num >= maxWorkingHour * intervalDuration:
#                 return False
#             return True
#         if node == 1 and one_path[i+1] == 0:
#             if num >= maxWorkingHour * intervalDuration:
#                 return False
#             num = 0

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

binary_N_paths = np.array([
    [0, 1, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 0],
    [1, 1, 0, 1, 1, 1, 0]
])

maxWorkingHour = 1
intervalDuration = 0.5
# print(maxWorkingHour / intervalDuration)
print(max_working_hour_constraint(binary_N_paths))
