from filters.discrete_time_kf import DiscreteTimeKF
import numpy as np


class SageHusaAKF(DiscreteTimeKF):
    def __init__(self,
                 model,
                 p0,
                 r0,
                 q0,
                 t0,
                 delta_t,
                 q_mean=None,
                 r_mean=None):
        super().__init__(model, p0, r0, q0, t0, delta_t)
        self.__iter = 0
        if q_mean is None:
            self.q_mean = np.zeros(model.p)
        else:
            self.q_mean = q_mean.copy()
        if r_mean is None:
            self.r_mean = np.zeros(model.m)
        else:
            self.r_mean = r_mean.copy()
        self.x_prev = np.zeros(model.n)
        self.P_prev = np.zeros(shape=(model.n, model.n))

    def __next_iter(self):
        self.__iter += 1

    def predict(self):
        self.__next_iter()
        self.x_prev[:] = self.x
        self.P_prev[:] = self.P
        s = self.G @ self.q_mean
        self.x = self.F @ self.x + self.Psi @ self.u + self.G @ self.q_mean
        self.P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T

    def update(self, y=None):
        if y is None:
            y = self.y.copy()

        y_predict = self.H @ self.x
        eps = y - y_predict
        B = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(B)
        self.x = K @ (eps - self.r_mean) + self.x

        self.__adapt_R(eps)

        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        self.y = self.H @ self.x
        self.__adapt_Q(eps, K)

        # Обновление матриц модели
        self.t = self.t + self.delta_t
        self.F = self.model.evaluate_f(self.t)
        self.Psi = self.model.evaluate_psi(self.t)
        self.G = self.model.evaluate_g(self.t)
        self.H = self.model.evaluate_h(self.t + self.delta_t)

    def reset(self, p0, r0, q0, t0, delta_t):
        self.__init__(self.model, p0, r0, q0, t0, delta_t)

    def __adapt_R(self, eps):
        self.r_mean = 1 / (self.__iter + 1) * (self.__iter * self.r_mean + eps)

        eps_tilda = eps[:, np.newaxis] - self.r_mean
        self.R = 1 / self.__iter * ((self.__iter - 1) * self.R + eps_tilda @ eps_tilda.T - self.H @ self.P @ self.H.T)

    def __adapt_Q(self, eps, K):
        gamma = np.linalg.inv(self.G.T @ self.G) @ self.G.T
        self.q_mean = 1 / self.__iter * ((self.__iter - 1) * self.q_mean + gamma @ (self.x - self.F @ self.x_prev))

        eps_tilda = eps[:, np.newaxis] - self.r_mean
        s = gamma @ (K @ eps_tilda @ eps_tilda.T @ K.T + self.P @ self.F @ self.P_prev @ self.F.T) @ gamma.T
        self.Q = 1 / self.__iter * ((self.__iter - 1) * self.Q + gamma @ (K @ eps_tilda @ eps_tilda.T @ K.T + self.P - self.F @ self.P_prev @ self.F.T) @ gamma.T)


