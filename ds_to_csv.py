'''
Author: Luke Bernier
Date: 4/7/21

This file converts data from data from machine learning algorithm to .csv file for processing
'''
import pandas as pd
import os

hl_size = []
speed_noise_val = []
dir_noise_val = []
number_of_times = []
names = []
accuracy = []
pecks_sum = []
sum_pecks_actual = []

directory = 'C:\\CCL\\neural networking\\nnsaves1\\'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    if os.path.isfile(f) and filename.endswith('.txt'):
        with open(f, 'r') as file:
            file.readline()
            number_of_times.append(int(file.readline()))
            hl_size.append(int(file.readline()))
            speed_noise_val.append(int(file.readline()))
            dir_noise_val.append(int(file.readline()))
            accuracy.append(float(file.readline()))
            pecks_sum.append(int(file.readline()))
            sum_pecks_actual.append(int(file.readline()))
            names.append(filename)

df = pd.DataFrame({'name': names, 'hl_size': hl_size, 'num_times': number_of_times, \
                   'speed_nv': speed_noise_val, 'dir_nv': dir_noise_val, 'acc': accuracy, \
                   'pecks_sum': pecks_sum, 'pecks_actual': sum_pecks_actual})

df.to_csv('C:\\CCL\\neural networking\\ds.csv', index=False)
