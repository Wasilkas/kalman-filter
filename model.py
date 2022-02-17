import numpy as np


def evaluate_f(x, t=None, u=None):
    return np.array([[x[0] + x[1], x[1]]])


def evaluate_g(t=None):
    return np.array([[1, 0], [0, 1]])


def evaluate_h(x, t=None, u=None):
    return np.array([x[0]])


def evaluate_r(t=None):
    return np.array([1])


def evaluate_x0():
    return np.array([0, 0])


def evaluate_p0():
    return np.array([[1, 0], [0, 1]])


def evaluate_q(t=None):
    return np.array([[1, 0], [0, 1]])


