import numpy as np


def MSE(target, predictions):
    m = target.size
    mse = np.sum(np.square(target - predictions)) / m
    return mse
