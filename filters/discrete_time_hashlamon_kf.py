import numpy as np

from discrete_time_kf import *


class HashlamonDKF(DiscreteTimeKF):
    def __init__(self, n, m, p0, r0, q0, x0, f, g, h, b, u, e, w, Nq, Nr):
        super().__init__(n, m, p0, r0, q0, x0, f, g, h, b, u)
        self.e = e
        self.w = w
        self.Nq = Nq
        self.Nr = Nr

    def update(self, y=None):
        if y is None:
            y = self.y

        y_predict = self.H @ self.x
        a1 = (self.Nr - 1) / self.Nr
        self.e = a1 * self.e + 1 / self.Nr * y_predict

        s = 1 / self.Nr * self.H @ self.P @ self.H.T
        ss = 1 / (self.Nr - 1) * (y_predict - self.e)[:, np.newaxis]
        sss = (y_predict - self.e)[:, np.newaxis].T
        deltaR = ss @ sss - s
        self.R = np.diag(np.abs(np.diag(a1 * self.R + deltaR)))

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        x_prev = self.x[:]
        p_prev = self.P[:]
        self.x = K @ (y - y_predict) + self.x
        self.P = (np.eye(self.n) - K @ self.H) @ self.P

        w_temp = self.x - x_prev
        a2 = (self.Nq - 1) / self.Nq
        self.w = a2 * self.w + 1 / self.Nq * w_temp

        deltaQ = 1 / self.Nq * (self.P - self.F @ p_prev @ self.F.T) + 1 / (self.Nq - 1) * (w_temp - self.w)[:, np.newaxis] @ (w_temp - self.w)[:, np.newaxis].T
        self.Q = np.diag(np.abs(np.diag(a2 * self.Q + deltaQ)))


