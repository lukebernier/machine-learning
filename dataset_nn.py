'''
Author: Luke Bernier
Date: 4/19/22

This file creates a dataset for the machine learning algorithm. This data set will be analyzed
to learn how the network reacts with different inputs.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from datetime import date
from datetime import datetime
import math


hidden_layer_size = 4
speed_noise_val = 5
dir_noise_val = 5
number_of_times = 5
number_of_sessions = 80

cur_date = date.today()
str_date = str(cur_date)
str_date.replace('-', '_')
print(str_date)
cur_time = datetime.now()
cur_time = cur_time.strftime("%H_%M_%S")

def output_session_data(num_sessions):
    '''
    :param num_sessions: number of sessions that are run to train the network
    :return: null
    '''
    cur_date = date.today()
    cur_time = datetime.now()
    cur_time = cur_time.strftime("%H:%M:%S")
    print('Running Neural Network\n')
    print('Date: ', cur_date)
    print('Time: ', cur_time)
    print('Number of sessions: ', num_sessions)


# FUNCTION NOT CURRENTLY IN USE
def output_test_data(model_output, X, Y, trial_num, session_num):

    print('Session: ', session_num + 1)
    print('Trial: ', trial_num + 1)
    print('inputs: ', X)
    print('expected output: ', Y)
    print('Model output: ', model_output)
    print('\n')


def run_sessions(num_times):
    '''

    :param num_times: number of times(sessions) to train the model on
    :return: null
    '''
    iter = 0
    cur_date = date.today()
    str_date = str(cur_date)
    str_date.replace('-', '_')
    print(str_date)
    cur_time = datetime.now()
    cur_time = cur_time.strftime("%H_%M_%S")
    for j in range(num_times):

        print('iter: ', iter, '/', num_times)
        iter += 1

        for k in range(80):

            X = session_vids[k]
            Y = torch.Tensor(new_Y[k]).view(-1, 1)
            for i in range(12):
                new_inputs = []
                if i == 0:
                    for l in range(len(X[i])):
                        new_inputs.append(X[i][l])
                    for l in range(6):
                        new_inputs.append(0)
                elif i == 1:
                    for l in range(len(X[i])):
                        new_inputs.append(X[i][l])
                    for l in range(3):
                        new_inputs.append(prev_inputs1[l])

                    for l in range(3):
                        new_inputs.append(0)
                else:
                    for l in range(len(X[i])):
                        new_inputs.append(X[i][l])
                    for l in range(3):
                        new_inputs.append(prev_inputs1[l])

                    for l in range(3):
                        new_inputs.append(prev_inputs2[l])

                new_inputs = torch.Tensor(new_inputs)

                inputs = Variable(new_inputs, requires_grad=False)
                targets = Variable(Y[i], requires_grad=False)

                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = model(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

                prev_inputs1 = []
                for m in range(3):
                    prev_inputs1.append(new_inputs[m])

                prev_inputs1 = torch.Tensor(prev_inputs1)

                prev_inputs2 = []
                if i == 0:
                    prev_inputs2 = torch.Tensor([0,0,0])
                else:
                    for m in (3,4,5):
                        prev_inputs2.append(new_inputs[m])

                    prev_inputs2 = torch.Tensor(prev_inputs2)


                prev_inputs1 = add_noise(prev_inputs1)
                prev_inputs2 = add_noise(prev_inputs2)

    filename = 'C:\\CCL\\neural networking\\nnsaves1\\' + 'nnsave' + str(cur_date) + str(cur_time) + '.pt'

    torch.save(model, filename)

def add_noise(val):
    '''

    :param val: array of len(3) -- each member of this array will have noise added to it
    based on the noise values defined prior
    :return: array of len(3) with modified values
    '''

    for i in range(len(val)):
        if i == 0:
            mean = 0
            sd = (val[i]*val[i])/speed_noise_val

            noise_val = np.random.normal(mean, sd)

            val[i] = val[i] + noise_val

            if val[i] < 0:
                val[i] = 0
            elif val[i] > 1:
                val[i] = 1
        elif i == 1:
            mean = 0
            sd = dir_noise_val
            noise_val = np.random.normal(mean, sd)
            noise_val = noise_val * (math.pi/180)
            val[i] = val[i].item()

            temp = val[i] + noise_val

            if (torch.le(temp, 1.0)) and (torch.ge(temp, -1.0)):
                val[i] = math.cos(math.acos(val[i] + noise_val))
            elif (torch.ge(temp, 1.0)):
                diff = (val[i] + noise_val) - 2
                val[i] = math.cos(math.acos(diff))
            else:
                diff = (val[i] + noise_val) + 2
                val[i] = math.cos(math.acos(diff))
        else:
            mean = 0
            sd = dir_noise_val
            noise_val = np.random.normal(mean, sd)
            noise_val = noise_val * (math.pi / 180)

            temp = val[i] + noise_val

            if (torch.le(temp, 1.0)) and (torch.ge(temp, -1.0)):
                val[i] = math.sin(math.asin(temp))
            elif (torch.ge(temp, 1.0)):
                diff = (temp) - 2
                val[i] = math.sin(math.asin(diff))
            else:
                diff = (temp) + 2
                val[i] = math.sin(math.asin(diff))
    return val

def create_same_videos():
    '''

    :return: two lists -- first list is len(12) -- each member of the list is a list of len(3) with values
    defining the video -- second list is len(12) defining which items the birds should peck at -- in
    this case, all 1's because it is a same video
    '''

    # TODO add a little noise due to perceptual noise
    new_arr = []

    available_dirs = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, \
                      225, 240, 255, 270, 285, 300, 315, 330, 345]
    available_speeds = [0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36, \
                        0.39, 0.42, 0.45, 0.48, 0.51, 0.54, 0.57, 0.60, 0.63, 0.66, 0.69, \
                        0.72, 0.75]

    baseline_dirs = [0, 30, 45, 75, 105, 120, 150, 180, 195, 225, 240, 255]
    baseline_speeds = [0.15, 0.18, 0.24, 0.30, 0.33, 0.39, 0.45, 0.48, 0.54, 0.6, 0.63, 0.66]

    speed = baseline_speeds.pop(np.random.randint(0, len(baseline_speeds)))
    dir = baseline_dirs.pop(np.random.randint(0, len(baseline_dirs)))

    dir = dir * (math.pi/180)
    dir_x = math.cos(dir)
    dir_y = math.sin(dir)

    for i in range(12):
        new_arr_x = [speed, dir_x, dir_y]
        new_arr.append(new_arr_x)

    new_arr_y = [1] * 12
    return(new_arr, new_arr_y)

def create_diff_videos():
    '''

    :return: two lists -- first list is len(12) -- each member of the list is a list of len(3) with values
    defining the video -- second list is len(12) defining which items the birds should peck at -- in
    this case, all 0's because it is a different video
    '''

    # TODO add a little noise due to perceptual noise

    new_arr = []

    available_dirs = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, \
                      225, 240, 255, 270, 285, 300, 315, 330, 345]
    available_speeds = [0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36, \
                        0.39, 0.42, 0.45, 0.48, 0.51, 0.54, 0.57, 0.60, 0.63, 0.66, 0.69, \
                        0.72, 0.75]

    baseline_dirs = [0, 30, 45, 75, 105, 120, 150, 180, 195, 225, 240, 255]
    baseline_speeds = [0.15, 0.18, 0.24, 0.30, 0.33, 0.39, 0.45, 0.48, 0.54, 0.6, 0.63, 0.66]

    for i in range(12):
        speed = baseline_speeds.pop(np.random.randint(0, len(baseline_speeds)))
        dir = baseline_dirs.pop(np.random.randint(0, len(baseline_dirs)))

        dir = dir * (math.pi / 180)

        dir_x = math.cos(dir)
        dir_y = math.sin(dir)

        new_arr_x = [speed, dir_x, dir_y]
        new_arr.append(new_arr_x)

    new_arr_y = [0] * 12

    return(new_arr, new_arr_y)



def create_session():
    '''

    :return: 3 items - 1: list of 80 trials - each member is either a same or different trial
    2: list of len(80) which defines which trials should be pecked at
    3: list of len(80) which defines which trials should be pecked at
    '''
    session_vids = []
    session_actuals = []

    peck_actuals = []

    for i in range(80):
        if i % 2 == 0:
            new_arr, new_arr_y = create_same_videos()
        else:
            new_arr, new_arr_y = create_diff_videos()

        session_vids.append(new_arr)
        session_actuals.append(new_arr_y)
        count = 0
        for j in range(len(new_arr_y)):
            if new_arr_y[j] == 0:
                count += 1
        if count >= 7:
            peck_actuals.append(0)
        else:
            peck_actuals.append(1)

    return(session_vids, session_actuals, peck_actuals)


 #model definition
class MLP(nn.Module):

    # define model elements
    def __init__(self):
        super(MLP, self).__init__()
        hl_size = hidden_layer_size
        self.lin1 = nn.Linear(9, hl_size)
        self.lin2 = nn.Linear(hl_size, 1)

    # forward propagate input
    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)

        x = self.lin2(x)
        return x

def test_model(peck_actuals):
    '''

    :param peck_actuals: list of which trials should be pecked at
    :return: accuracy: float: how many trials the model correctly responded to
    pecks_sum: int: used to compute accuracy
    sum_peck_actuals: int: total number of pecks
    '''
    predictions = []
    actuals = []

    peck_predictions = []
    pecks_sum = 0
    sum_peck_actuals = 0
    for i in range(len(peck_actuals)):
        if peck_actuals[i] == 1:
            sum_peck_actuals += 12

    for j in range(80):
        X = session_vids[j]
        Y = torch.Tensor(new_Y[j]).view(-1, 1)


        for i in range(12):
            new_inputs = []
            if i == 0:
                for l in range(len(X[i])):
                    new_inputs.append(X[i][l])
                for l in range(6):
                    new_inputs.append(0)
            elif i == 1:
                for l in range(len(X[i])):
                    new_inputs.append(X[i][l])
                for l in range(3):
                    new_inputs.append(prev_inputs1[l])

                for l in range(3):
                    new_inputs.append(0)
            else:
                for l in range(len(X[i])):
                    new_inputs.append(X[i][l])
                for l in range(3):
                    new_inputs.append(prev_inputs1[l])

                for l in range(3):
                    new_inputs.append(prev_inputs2[l])

            new_inputs = torch.Tensor(new_inputs)
            inputs = Variable(new_inputs, requires_grad=False)
            targets = Variable(Y[i], requires_grad=False)

            # evaluate the model on the test set
            yhat = model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            actual = actual.reshape((len(actual), 1))
            # round to class values
            yhat = yhat.round()
            # store
            predictions.append(yhat)
            actuals.append(actual)

            prev_inputs1 = new_inputs
            prev_inputs2 = prev_inputs1

            prev_inputs1 = add_noise(prev_inputs1)
            prev_inputs2 = add_noise(prev_inputs2)

        count = 0

        for k in range(len(predictions)):
            if predictions[k] == 0:
                count += 1
                pecks_sum += 1

        if count >= 7:
            peck_predictions.append(0)
        else:
            peck_predictions.append(1)

        predictions = []

    accuracy = 0
    for i in range(80):
        if peck_predictions[i] == peck_actuals[i]:
            accuracy += 1

    accuracy = accuracy / 80
    print('model accuracy: ', accuracy)

    return(accuracy, pecks_sum, sum_peck_actuals)

#create dataset
count = 1
for _ in range(10):
    for i in range(1, 13):
        hidden_layer_size = i

        for j in (3, 6, 9):
            speed_noise_val = j

            for k in (5, 10, 15):
                dir_noise_val = k

                for l in (10, 20, 30, 40, 50):
                    number_of_times = l
                    model = MLP()

                    cur_date = date.today()
                    str_date = str(cur_date)
                    str_date.replace('-', '_')
                    cur_time = datetime.now()
                    cur_time = cur_time.strftime("%H_%M_%S")

                    criterion = nn.MSELoss()
                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

                    session_vids, new_Y, peck_actuals = create_session()

                    X = session_vids[0]
                    Y = torch.Tensor(new_Y[0]).view(-1,1)

                    data_file_name = 'C:\\CCL\\neural networking\\nnsaves1\\' + 'data' + str_date + str(cur_time) + '.txt'

                    print(str(count) + '/5400')
                    count += 1

                    output_session_data(number_of_sessions)
                    run_sessions(number_of_times)
                    accuracy, pecks_sum, sum_peck_actuals = test_model(peck_actuals)

                    with open(data_file_name, 'w') as file:
                        file.write(str(number_of_sessions))
                        file.write('\n')
                        file.write(str(number_of_times))
                        file.write('\n')
                        file.write(str(hidden_layer_size))
                        file.write('\n')
                        file.write(str(speed_noise_val))
                        file.write('\n')
                        file.write(str(dir_noise_val))
                        file.write('\n')
                        file.write(str(accuracy))
                        file.write('\n')
                        file.write(str(pecks_sum))
                        file.write('\n')
                        file.write(str(sum_peck_actuals))
                        file.write('\n')
                        for i in range(len(session_vids)):
                            file.write(str(session_vids[i]))
                            file.write('\n')
                        for i in range(len(new_Y)):
                            file.write(str(new_Y[i]))
                            file.write('\n')
                        file.write(str(peck_actuals))