"""
@author: Vincent Bonnet
@description : Polynomial Linear Regression on 1D data set
"""

import numpy as np
import matplotlib.pyplot as plt

def FUNCTION_1D(x):
    return np.sin(x) * np.cos((x+1)*1.1) * 2
MIN_RANGE = 0.0
MAX_RANGE = 10.0
NUM_SAMPLES = 20
POLYNOMIAL_DEGREE = 11

def random_sample_from_function_1D(function, min_range, max_range, num_samples):
    samples = np.zeros((num_samples, 2))
    samples_x = np.linspace(min_range, max_range, num_samples, endpoint=True)
    #random_offset_y = (np.random.rand(num_samples) * 0.5) - 0.25 # for jittering
    random_offset_y = np.zeros(num_samples)

    for i in range(0, num_samples):
        samples[i] = (samples_x[i], function(samples_x[i]) + random_offset_y[i])

    return samples

'''
 Drawing Methods
'''
def draw_function_1D(ax, function, min_range, max_range, draw_step):
    t = np.linspace(min_range, max_range, int((max_range - min_range) / draw_step), endpoint=True)
    plt.plot(t, function(t), '-', color='green', label='Reference function')

def draw_samples(ax, samples):
    x, y = zip(*samples)
    plt.plot(x, y, '.', color='red', label='Samples')

def draw_poly_function(ax, poly_weights, min_range, max_range, draw_step):
    x = np.linspace(min_range, max_range, int((max_range - min_range) / draw_step), endpoint=True)
    poly_degree = np.size(poly_weights)
    num_samples = np.size(x)

    y = np.zeros(num_samples)
    for i in range(num_samples):
        for exponent_id in range(poly_degree):
            y[i] += (x[i]**(exponent_id) * poly_weights[exponent_id])

    plt.plot(x, y, '-', linestyle='dotted', color='blue', label='Polynomial regression')

'''
Polynomial Regression
Solves y = Xb + error where
y is the vector of observed values [y0, y1, y2, ...] of size(num sample,1)
X is the Vandermonde matrix of size(num_sample, poly_degree)
|1 x0 x0^2 x0^3 .|
|1 x1 x1^2 x1^3 .|
|1 x2 x2^2 x2^3 .|
|. .. .... .... .|
b is the unknown weights [b0, b1, b2, ...] (poly_degree, 1)
e is the error vector [e0, e1, e2, ...] (num_sample, 1)
'''
def polynomial_regression_weights(samples, poly_degree):
    sample_x, sample_y = zip(*samples)
    num_sample = np.size(sample_y)

    y = np.reshape(np.asarray(sample_y), (num_sample,1))
    X = np.matrix(np.vander(sample_x, poly_degree, increasing=True))

    # Solve with the pseudo inverse
    pseudo_inverse = np.linalg.inv(X.transpose() * X)
    pseudo_inverse = pseudo_inverse * X.transpose()

    b = np.matmul(pseudo_inverse, y)

    return b

def main():
    fig, ax = plt.subplots()
    ax.grid()
    ax.axis('equal')

    samples = random_sample_from_function_1D(FUNCTION_1D, MIN_RANGE, MAX_RANGE, NUM_SAMPLES)
    draw_function_1D(ax, FUNCTION_1D, MIN_RANGE, MAX_RANGE, draw_step = 0.1)
    draw_samples(ax, samples)

    poly_weights = polynomial_regression_weights(samples, POLYNOMIAL_DEGREE)

    sample_x, unused_y = zip(*samples)
    draw_poly_function(ax, poly_weights, MIN_RANGE, MAX_RANGE, draw_step = 0.1)

    # display
    plt.title('Polynomial Regression')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.show()

if __name__ == '__main__':
    main()
