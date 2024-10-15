import numpy as np

from activations import *


# TODO: initialisation options
class Node:
    def __init__(
        self, name=None, input_size=None, weights=None, bias=0, activation_function=ReLu
    ):
        self.name = name
        self.input_size = input_size
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.activation = 0
        self.out = 0
        self.delta = 0  # error message used for backpropagation

    def forward(self, x):
        if self.input_size > 1:
            self.activation = np.dot(x, self.weights) + self.bias
        else:
            self.activation = x * self.weights[0] + self.bias

        self.out = self.activation_function(self.activation)
        return self.out
