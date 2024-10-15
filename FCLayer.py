import numpy as np

from activations import *
from loss_functions import *
from Node import Node


class FCLayer:
    def __init__(
        self,
        layer_size,
        input_size,
        activation_function=ReLu,
        bias=0,
        initialisation="Gaussian",
    ):
        self.layer_size = layer_size
        self.input_size = input_size
        self.nodes = np.empty(self.layer_size, dtype=Node)
        self.deltas = np.zeros(self.layer_size)

        # Activation function and derivative for backpropagation
        self.activation_function = activation_function
        self.dot_function = activation_functions.get(self.activation_function)

        # Activation values (of each node)
        self.inputs = np.zeros(self.layer_size)
        # Output values (of each node)
        self.out_values = np.empty(self.layer_size)

        self.dL_dw = np.zeros((self.input_size, self.layer_size))
        self.dL_db = np.zeros(self.layer_size)

        # for SGD
        print(f"Layer: input size = {self.input_size}, layer size = {self.layer_size}")

        self.dL_dw_cumulative = np.zeros((self.input_size, self.layer_size))
        self.dL_db_cumulative = np.zeros(self.layer_size)

        self.initialisation = initialisation

        for i in range(self.layer_size):
            self.nodes[i] = Node(
                name=f"node {i}",
                input_size=input_size,
                activation_function=activation_function,
                weights=self.initialiser(),
                bias=bias,
            )

    def zero_grad(self):
        self.dL_dw_cumulative = np.zeros((self.input_size, self.layer_size))
        self.dL_db_cumulative = np.zeros(self.layer_size)

    def initialiser(self):
        if self.initialisation == "Gaussian":
            std_dev = 0.5
            return np.random.normal(0, std_dev, self.input_size)

    def forward(self, x):
        self.inputs = x
        self.out_values = np.array([node.forward(x) for node in self.nodes])
        return self.out_values

    def __call__(self, x):
        return self.forward(x)

    def print_weights(self):
        print(f"Layer Nodes ({self.layer_size})")
        for node in self.nodes:
            print(f"{node.name}, Weights: {node.weights}, Bias: {node.bias}")

    def backpropagate(self, next_layer=None, cost_function=MSE, target=None):
        # If the layer is the last:
        if next_layer is None:
            dL_do = -(target - self.out_values)

            for i in range(self.layer_size):
                self.deltas[i] = dL_do[i] * self.dot_function(self.nodes[i].activation)

        else:
            next_deltas = next_layer.deltas
            for i in range(self.layer_size):
                weights = np.array([node.weights[i] for node in next_layer.nodes])
                self.deltas[i] = self.dot_function(self.nodes[i].activation) * np.dot(
                    weights, next_deltas
                )

        # # Compute weights gradients
        self.dL_dw = np.outer(self.inputs, self.deltas)

        # Compute bias gradients
        self.dL_db = self.deltas

        # Sum gradients for SGD
        self.dL_dw_cumulative += self.dL_dw
        self.dL_db_cumulative += self.dL_db

    def update_weights(self, learning_rate=0.1, batch_size=1):
        # Compute weights gradients

        dL_dw = self.dL_dw_cumulative / batch_size if batch_size > 1 else self.dL_dw
        dL_db = self.dL_db_cumulative / batch_size if batch_size > 1 else self.dL_db

        for i, node in enumerate(self.nodes):
            node.weights = node.weights - learning_rate * dL_dw[:, i]

        for i, node in enumerate(self.nodes):
            node.bias = node.bias - learning_rate * dL_db[i]
