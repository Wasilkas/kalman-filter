from models.model import Model
import numpy as np


class GyroscopeModel(Model):
    n = 2
    m = 1
    r = 1
    p = 2

    def __init__(self, deltaT=0.01):
        super().__init__()
        self.deltaT = 0.1

    def evaluate_f(self, t=None):
        return np.array([[1, -self.deltaT],
                         [0, 1]])

    @staticmethod
    def x2(t):
        if 0 <= t <= 20:
            return 0.5
        elif 20 < t <= 50:
            return 1
        else:
            return 0

    def evaluate_psi(self, t=None):
        return np.array([[self.deltaT],
                        [0]])

    def evaluate_g(self, t=None):
        return np.array([[1, 0],
                         [0, 1]])

    def evaluate_u(self, t=None):
        return np.array([1])

    def evaluate_h(self, t=None):
        return np.array([[1, 0], ])

    def evaluate_r(self, t=None):
        r0 = 0.01
        return np.array([[r0], ])

    def evaluate_x0(self):
        return np.array([0, 0])

    def evaluate_q(self, t=None):
        return np.array([[1e-4, 0],
                        [0, 1e-6]])
