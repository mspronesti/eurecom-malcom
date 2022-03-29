import numpy as np


def relu_naive(x: np.ndarray):
    """Naive Rectified linear activation"""
    x[x <= 0] = 0
    return x


def relu(x: np.ndarray):
    """Better Rectified linear activation"""
    # this basically follows the definition
    # of max(0, x)
    return np.maximum(0, x)


def softmax(x: np.ndarray):
    """
    Softmax activation when classes are on rows,
    i.e. when X.shape == (N,C), being N the number of samples
    and C the number of classes
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
