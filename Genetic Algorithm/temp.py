import numpy as np

def max_working_hour_constraint(binary_N_paths):
    '''
    make sure that no driver works more than a few hours continuously
    '''
    violationCount = 0
    for one_path in binary_N_paths:
        num, num_list = 0, []
        one_path_copy = one_path.copy()
        # first check if rush hour 10 actually is 11.
        if checkRushHourFlag:
            if one_path_copy[1] + one_path_copy[2] == 1:
                one_path_copy[1] = 1
                one_path_copy[2] = 1
            # if one_path_copy[21] + one_path_copy[22] == 1:
            #     one_path_copy[21] = 1
            #     one_path_copy[22] = 1
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

checkRushHourFlag = True
maxWorkingHour = 4
intervalDuration = 0.5

binary_N_paths = np.array([
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
])
print('original:', binary_N_paths)
print(max_working_hour_constraint(binary_N_paths))
print('   after:', binary_N_paths)
