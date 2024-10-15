import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dot_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def identity(x):
    return x


def dot_identity(x):
    return 1


def tanh(x):
    return np.tanh(x)


def dot_tanh(x):
    return 1 - np.square(np.tanh(x))


def ReLu(x):
    return np.max(x, 0)


def dot_ReLu(x):
    return 0 if x < 0 else 1


def LReLu(x, alpha=0.3):
    return x if x > 0 else alpha * x


def dot_LReLu(x, alpha=0.3):
    return 1 if x >= 0 else alpha


def ELU(x, alpha=1):
    return x if x >= 0 else alpha * np.exp(x - 1)


def dot_ELU(x, alpha=1):
    return 1 if x >= 0 else alpha * np.exp(x - 1)


activation_functions = {
    sigmoid: dot_sigmoid,
    tanh: dot_tanh,
    identity: dot_identity,
    ReLu: dot_ReLu,
    LReLu: dot_LReLu,
    ELU: dot_ELU,
}
