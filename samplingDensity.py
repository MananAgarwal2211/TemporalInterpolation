from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import lagrange
import pandas as pd
import numpy as np

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
data = read.to_numpy()

final = []
for i in range(2, 11): # i varies the sampling density

    max = int(99 / i) * i + 1
    source_data = data[:max]

    # manipulating sampling density of training set
    samples = source_data[::i].copy()

    # splitting training x and y
    sample_x = samples[:, 0]
    sample_y = samples[:, 1]

    # setting remaining data entries as tests
    testing_points = np.delete(source_data[:, 0], slice(None, None, i))
    testing_values = np.delete(source_data[:, 1], slice(None, None, i))

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
        error = abs(intpl - actl)
        relerror = 100 * error / actl
        rel_error.append(relerror)
    final.append(sum(rel_error) / len(rel_error))

# output (may be logged into spreadsheet)
for item in final:
    print(item)