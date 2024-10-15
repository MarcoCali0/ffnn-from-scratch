import numpy as np

from activations import *
from FCLayer import FCLayer
from loss_functions import *


class FFNN:
    def __init__(self):
        self.layers = []

    def predict(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

    def backpropagate(self, target, cost_function=MSE, batch_size=1):
        # Backpropagate through the layers in reverse order
        N = len(self.layers)

        if batch_size > 1:
            self.layers[-1].backpropagate(target=target)
            for i in range(N - 2, -1, -1):
                self.layers[i].backpropagate(next_layer=self.layers[i + 1])
        else:
            self.layers[-1].backpropagate(target=target)
            for i in range(N - 2, -1, -1):
                self.layers[i].backpropagate(next_layer=self.layers[i + 1])

    def SGD_step(self, learning_rate=0.1, batch_size=1):
        for layer in self.layers:
            layer.update_weights(learning_rate, batch_size=batch_size)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()