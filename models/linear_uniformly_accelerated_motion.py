import numpy as np
import models.model


# Задача Чубич Вестник ИрГТУ(2017) без управления
# Linear uniformly accelerated motion
# Поступательное равноускоренное движение
class LinUniAccMotion(models.model.Model):
    n = 3
    m = 1
    r = 1
    p = 3

    def __init__(self, deltaT=0.1):
        super().__init__()
        self.deltaT = deltaT

    def evaluate_f(self, t=None):
        return np.array([[1, self.deltaT, self.deltaT ** 2 / 2],
                         [0, 1, self.deltaT],
                         [0, 0, 1]])

    def evaluate_psi(self, t=None):
        return np.array([[0],
                         [0],
                         [0]])

    def evaluate_u(self, t=None):
        return np.array([0])

    def evaluate_g(self, t=None):
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    def evaluate_h(self, t=None):
        return np.array([[0, 1, 0], ])

    def evaluate_r(self, t=None):
        r0 = 0.01
        return np.array([[r0], ])

    def evaluate_x0(self):
        return np.array([0, 0, 1])

    def evaluate_q(self, t=None):
        q0 = 0.01
        return np.array([[q0, 0, 0],
                         [0, q0, 0],
                         [0, 0, q0]])
