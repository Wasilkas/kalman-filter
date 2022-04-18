import math
import numpy as np
import models.model


# Задача Чубич(2017) с управлением (Вестник ИрГТУ)
# Модель системы контроля положения
class SystemPositionControl(models.model.Model):
    n = 2
    m = 1
    r = 1
    p = 1

    def __init__(self, T=0.1, theta1=4.6, theta2=0.787):
        super().__init__()
        self.T = T
        self.theta1 = theta1
        self.theta2 = theta2

    def evaluate_psi(self, t=None):
        return np.array([[self.theta2/self.theta1*(self.T-1/self.theta1+math.exp(-self.theta1*self.T)/self.theta1)],
                         [self.theta2/self.theta1*(1-math.exp(-self.theta1*self.T))]])

    def evaluate_u(self, t=None):
        return np.array([12])

    def evaluate_f(self, t=None):
        return np.array([[1, 1/self.theta1*(1-math.exp(-self.theta1*self.T))],
                         [0, math.exp(-self.theta1*self.T)]])

    def evaluate_g(self, t=None, isEqualDimensions=False):
        if isEqualDimensions:
            return np.array([[1, 0],
                            [0, 1]])
        else:
            return np.array([[0],
                            [1]])

    def evaluate_h(self, t=None):
        return np.array([[1, 0], ])

    def evaluate_r(self, t=None):
        r0 = 0.1
        return np.array([[r0], ])

    def evaluate_x0(self):
        return np.array([0, 0])

    def evaluate_q(self, t=None, isEqualDimensions=False):
        q0 = 0.01
        if isEqualDimensions:
            return np.array([[q0, 0],
                            [0, q0]])
        else:
            return np.array([[q0]])
