'''
Implementation of simple neural network which contains only one hidden layer
@Author: Rohith Uppala
'''

import numpy as np
import math

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        '''
        Constructor is used to initialize the input, output weights
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = int(math.sqrt(input_size + output_size))
        # Initializing the required variables
        self.W  = np.random.uniform(-np.sqrt(input_size), np.sqrt(input_size), (input_size, hidden_size))
        self.W1 = np.random.uniform(-np.sqrt(input_size), np.sqrt(input_size), (output_size, hidden_size))
        self.h = np.zeros(hidden_size)
        self.o = np.zeros(output_size)

    # Required Utility functions

    def tanh(self, x):
        '''
        This method will calculate the tanh. tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        '''
        e_2x = 2 * np.exp(x - np.max(x))
        return (e_2x - 1) * 1.0 / (e_2x + 1)

    def softmax(self, x):
        '''
        This method will implement the softmax function
        s(x) = e^x / sigma(e^xi)
        '''
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)


    def forward_propogation(self, x):
        '''
        This method will compute forward propogation of network
        '''
        h = self.tanh(self.W.dot(x))
        o = self.softmax(self.W1.dot(h))
        return [o, h]

    def backward_propogation(self, x, y):
        '''
        This method will compute backward propogation of network
        '''
        o, h = self.forward_propogation(x)
        delta_W1 = np.zeros(self.W1.shape)
        delta_W = np.zeros(self.W.shape)

        # Let's calculate the delta for ouput weghts W1
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                delta_W1[j][k] += 2 * (y[k] - o[k]) * o[k] * (1 - o[k]) * h[j]

        # Let's calculate the delta for input weights W
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                for k in range(self.output_size):
                    delta_W[i][j] += 2 * (y[k] - o[k]) * o[k] * (1 - o[k]) * delta_W1[j][k] * (1 - (h[j] * h[j])) * x[i]

        return [delta_W1, delta_W]
