from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import lagrange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
original = pd.read_csv('DataMM.csv')

values = []
for k in range(1, 11): # sampling range varies with k
    final = []
    for j in range(10):
        read = original[1:(10 * k) - 1] # excluding endpoints
        read = read.iloc[np.random.permutation(len(read))] # randomising order of data entries
        read.reset_index(drop=True)

        data = read[:7 * k - 2].to_numpy() # allocating training entries
        samples = data[data[:, 0].argsort()] # sorting training entries by x-values

        # adding back x-endpoints
        sample_points = samples[:, 0].tolist()
        sample_points.insert(0, 1)
        sample_points.append(10 * k)
        sample_x = np.array(sample_points)

        # adding back y-endpoints
        sample_values = samples[:, 1].tolist()
        sample_values.insert(0, 769)
        sample_values.append(original.to_numpy()[10 * k - 1, 1])
        sample_y = np.array(sample_values)

        # setting testing data
        testing_points = read.to_numpy()[7 * k - 2:10 * k, 0]
        testing_values = read.to_numpy()[7 * k - 2:10 * k, 1]

        # defining interpolation functions
        f = interp1d(sample_x, sample_y, kind="linear")
        g = CubicSpline(sample_x, sample_y, bc_type="natural")
        h = idw(sample_x, sample_y)

        # running function of choice
        estimations = f(testing_points)

        # crossvalidation and RME calculation
        rel_error = []
        for i in range(len(testing_values)):
            intpl = estimations[i]
            actl = testing_values[i]
            error = abs(intpl - actl)
            relerror = 100 * error / actl
            rel_error.append(relerror)
        final.append(sum(rel_error) / len(rel_error))

    values.append(sum(final)/len(final))

# output (may be logged into spreadsheet)
for item in values:
    print(item)