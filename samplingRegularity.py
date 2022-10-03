from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import lagrange
import pandas as pd
import numpy as np
import random

# generating data entry indices with prescribed regularity (1 to 10)
def generator(k):
    list = [k]*20
    i=0
    while (i < (100 - 20*k)):
        index = random.randint(0, 19)
        if list[index] < (10-k):
            list[index] += 1
            i += 1
    index = 0
    indices = []
    for i in range(len(list)):
        indices.append(index)
        index += list[i]
    indices.append(index)
    return indices

def idw(x, y):
    def final(testing):
        weights = []
        for i in x:
            weight = 1 / abs(testing - i)
            weights.append(weight)
        denominator = sum(weights)
        numerator = 0
        for j in range(len(y)):
            numerator += weights[j] * y[j]
        return (numerator / denominator)
    return final

# inputting data
read = pd.read_csv('DataMM.csv')
source_data = read.to_numpy()

final = []
for k in range(1, 11):
    trial_errors = []
    for j in range(1):

        indices = generator(k) # # generating data entry indices with prescribed regularity (1 to 10)

        # setting training data
        samples = source_data[indices]
        sample_x = samples[:, 0]
        sample_y = samples[:, 1]

        # setting testing data
        testing_points = np.delete(source_data[:, 0], indices)
        testing_values = np.delete(source_data[:, 1], indices)

        # defining interpolation functions
        f = interp1d(sample_x, sample_y, kind="linear")
        g = CubicSpline(sample_x, sample_y, bc_type="natural")
        h = idw(sample_x, sample_y)

        # running function of choice
        estimations = h(testing_points)

        # crossvalidation and RME calculation
        rel_error = []
        for i in range(len(testing_values)):
            intpl = estimations[i]
            actl = testing_values[i]
            error = abs(intpl-actl)
            relerror = 100 * error / actl
            rel_error.append(relerror)
        trial_errors.append(sum(rel_error) / len(rel_error))

    final.append(sum(trial_errors) / len(trial_errors))

# output (may be logged into spreadsheet)
for item in final:
    print(item)