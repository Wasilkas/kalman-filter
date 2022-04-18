from filters.discrete_time_kf import DiscreteTimeKF
import numpy as np


# Для использования данной модификации нужно привести компоненты шума в уравнении состояний
# к размеру вектора-состояний (nxn, nx1)
class YuanLeiAdaptiveDKF(DiscreteTimeKF):
    forget_factor = 0.95

    def __init__(self, n, m, p0, r0, q0, x0, f, g, h, b, u, q_mean, r_mean):
        super().__init__(n, m, p0, r0, q0, x0, f, g, h, b, u)
        self.iter = 1
        self.x_amendatory = x0
        self.q_mean = q_mean
        self.r_mean = r_mean
        self.x_prev = np.zeros(n)

    def predict(self):
        self.x_prev[:] = self.x
        self.x = self.F @ self.x + self.B @ self.u + self.q_mean
        self.P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T

    def update(self, y=None):
        if y is None:
            y = self.y
        d = (1 - self.forget_factor) / (1 - self.forget_factor ** self.iter)

        y_predict = self.H @ self.x
        Y = y - y_predict - self.r_mean
        self.R = (1 - d) * self.R + d * (Y @ Y.T - self.H @ self.P @ self.H.T)
        self.r_mean = (1 - d) * self.r_mean + d * (y - y_predict)

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = K @ Y + self.x
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        self.Q = (1 - d) * self.Q + d * (K @ Y[:, np.newaxis] @ Y.T[:, np.newaxis] @ K.T + self.P - self.F @ self.P @ self.F.T)
        self.q_mean = (1 - d) * self.q_mean + d * (self.x - self.F @ self.x_prev)
        self.iter += 1
