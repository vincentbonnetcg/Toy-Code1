"""
@author: Vincent Bonnet
@description : Neural network utilities
"""

'''
#CONVENTIONS#

num_inputs : number of inputs on the logistic regression
num_examples : number of examples used for the training
|Array Name   |  Array Shape               | Description                      |
|-------------|----------------------------|----------------------------------|
|w            | (num_inputs, 1)            | input weights                    |
|b            | (1,1)                      | input bias                       |
|y_hat        | (1, num_examples)          | activation per examples          |
|y            | (1, num_examples)          |                                  |
|X            | (num_inputs, num_examples) | training data stored into column |

|Function Name | Description         |
|--------------|---------------------|
|sigma         | activation function |
|L(y_hat, y)   | error function      |
'''

import numpy as np
import matplotlib.pyplot as plt

def sigmoid_activation(z):
    # TODO - add other activation function such as tanh and ReLU
    '''
    Sigmoid activation Function
    '''
    return 1.0 / (1.0 + np.exp(-z))

def draw_activation(f):
    '''
    Draw sigmoid function
    '''
    x = np.linspace(-10.0, 10.0, num=100, endpoint=True)
    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, f(x), 'r-')

def cross_entropy_cost_function(y, y_hat):
    '''
    'y' is the value from the test set
    'y_hat' is the value from the activation function
    '''
    m = y.shape[1]
    L = -(1./m) * (np.sum(np.multiply(y, np.log(y_hat))) + np.sum(np.multiply((1-y), np.log(1-y_hat))))
    return L

class LogisticRegressionMNIST():
    '''
    Logistic regression to detect a single number
    Aka Neural Network with a single unit
    '''
    def __init__(self, number_to_detect):
        # Value to detect
        self.number_to_detect = number_to_detect

        # Parameters
        self.X = None # training data with a shape (num_inputs, num_examples)
        self.w = None # input weights with a shape (num_inputs, 1)

        # Optimizer Hyperparameters
        self.learning_rate = 1.0
        self.minibatch_size = 1 # Not used
        self.num_epoch = 1000

        # Model Hyperparameters (Unused)
        # Only for future when becomes a fully fonctional neural network
        #self.num_layers = 1
        #self.num_units = 1

    def train(self, training_data_array):
        '''
        Compute the input weights for a single logistic regression
        '''
        # Prepare training data and parameters
        self.X = training_data_array.transpose()
        num_inputs = len(self.X) # self.X.shape[0]
        num_examples = self.X.shape[1]
        self.w = np.random.randn(num_inputs, 1)
        print(num_inputs)

        # Train
        for epoch_id in range(self.num_epoch):
            # TODO
            pass

    def forward_propagation(self):
        '''
        Forward propagation
        '''
        pass


    def back_propagation(self):
        '''
        Figure out how the cost function
        '''
        pass

