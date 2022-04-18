import numpy as np
import models.model


class DiscreteTimeKF:
    def __init__(self, model, p0, r0, q0, t0, delta_t):
        self.model = model
        self.t = t0
        self.delta_t = delta_t
        self.n = model.n
        self.m = model.m
        self.P = p0.copy()
        self.R = r0.copy()
        self.Q = q0.copy()
        self.x = model.evaluate_x0()
        self.y = np.zeros(shape=(self.m, 1,))
        self.F = model.evaluate_f(t0)
        self.G = model.evaluate_g(t0)
        self.H = model.evaluate_h(t0 + delta_t)
        self.Psi = model.evaluate_psi(t0)
        self.u = model.evaluate_u(t0)
        self.__iter = 0

    def __next_iter(self):
        self.__iter += 1

    def predict(self):
        self.__next_iter()
        self.x = self.F @ self.x + self.Psi @ self.u
        self.P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T

    def update(self, y=None):
        if y is None:
            y = self.y.copy()

        y_predict = self.H @ self.x
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = K @ (y - y_predict) + self.x
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        self.y = self.H @ self.x

        # Обновление матриц модели
        self.t = self.t + self.delta_t
        self.F = self.model.evaluate_f(self.t)
        self.Psi = self.model.evaluate_psi(self.t)
        self.G = self.model.evaluate_g(self.t)
        self.H = self.model.evaluate_h(self.t + self.delta_t)

    def reset(self, p0, r0, q0, t0, delta_t):
        self.__init__(self.model, p0, r0, q0, t0, delta_t)
