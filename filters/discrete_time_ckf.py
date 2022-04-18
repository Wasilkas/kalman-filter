import numpy.linalg as npl
from models import *


class DiscreteTimeCKF:
    def __init__(self, n, m, p0, r0, q0, x0):
        self.n = n
        self.m = m
        self.p = p0
        self.r = r0
        self.q = q0
        self.x = x0
        self.xi = np.zeros(shape=(2*n, n))
        self.y = np.zeros(shape=(m, 1))

        for i in range(self.n):
            sqrt_n = math.sqrt(float(self.n))
            self.xi[i][i] = sqrt_n
            self.xi[self.n + i][i] = -sqrt_n

    def predict(self):
        chi_predict = np.zeros(shape=(2*self.n, self.n))
        chol_p = npl.cholesky(self.p)
        x_prev = self.x[:]
        self.x = np.zeros(shape=(self.n, 1))
        self.p = np.zeros(shape=(self.n, self.n))

        for i in range(2 * self.n):
            chi_predict[i] = evaluate_f(x_prev + chol_p @ np.transpose(self.xi[i]))
            self.x += chi_predict[i] / (2 * self.n)

        for i in range(2 * self.n):
            eps = np.reshape(chi_predict[i] - self.x, (self.n, 1))
            self.p += eps @ np.transpose(eps)
        self.p /= 2
        g = evaluate_g()
        self.p += g() @ evaluate_g() @ np.transpose(g)

    def update(self, y=None):
        if y is None:
            y = self.y

        chi = np.zeros(shape=(2*self.n, self.n))
        gamma = np.zeros(shape=(2*self.n, self.m))
        p_yy = np.zeros(shape=(self.m, self.m))
        p_xy = np.zeros(shape=(self.n, self.m))
        y_pred = np.zeros(self.m)
        chol_p = npl.cholesky(self.p)

        for i in range(2*self.n):
            chi[i] = self.x + chol_p @ np.transpose(self.xi[i])
            gamma[i] = evaluate_h(chi[i])
            y_pred += gamma[i]
        y_pred /= 2 * self.n

        for i in range(0, 2*self.n):
            chi_x = np.reshape(chi[i] - self.x, (self.n, 1))
            gamma_y = np.reshape(gamma[i] - y_pred, (self.m, 1))
            p_yy += gamma_y @ np.transpose(gamma_y)
            p_xy += chi_x @ np.transpose(gamma_y)
        p_yy /= 2 * self.n
        p_yy += evaluate_r()
        p_xy /= 2

        k = p_xy @ npl.pinv(p_yy)
        self.x += k @ (y - y_pred)
        self.p += k @ p_yy @ np.transpose(k)
        self.y = evaluate_h(self.x)
