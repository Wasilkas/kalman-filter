from filters.discrete_time_kf import DiscreteTimeKF
import numpy as np


class DiscreteTimeGaoAKF(DiscreteTimeKF):
    def __init__(self,
                 n,
                 m,
                 p0,
                 r0, q0,
                 x0,
                 f, g,
                 h,
                 b,
                 u,
                 q_mean,
                 r_mean,
                 moving_window=10,
                 iter=0):
        super().__init__(n, m, p0, r0, q0, x0, f, g, h, b, u)
        self.q_mean = q_mean
        self.r_mean = r_mean
        self.__iter = iter
        self.__fading_factor = 0.7
        self.moving_window = moving_window
        self.C = np.zeros(shape=(m, m))
        self.__eps_N = np.zeros(shape=(moving_window, m))
        self.__eps_curr = np.zeros(m)
        self.P_prev = np.zeros(shape=(n, n))

    def __next_iter(self):
        self.__iter += 1

    def one_step_smooth(self, y=None):
        self.__next_iter()
        if y is None:
            y = self.y[:]

        x_temp = self.F @ self.x + self.B @ self.u + self.G @ self.q_mean
        eps_temp = y - self.H @ x_temp - self.r_mean
        self.P_prev[:] = self.P

        self.P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        K = self.P_prev @ self.F.T @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ eps_temp

    def predict(self):
        self.x = self.F @ self.x + self.B @ self.u + self.G @ self.q_mean

    def update(self, y=None):
        P_pred = np.zeros(shape=(self.n, self.n))
        if y is None:
            y = np.zeros(self.m)
            y[:] = self.y

        eps = y - self.H @ self.x - self.r_mean
        self.__eps_curr[:] = eps

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ eps
        P_pred[:] = self.P
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        self.__adapt(K, P_pred)

    def __adapt(self, K, P_pred):
        xi = np.linalg.inv(self.G.T @ self.G) @ self.G.T
        tau = (1 - self.__fading_factor) / (1 - self.__fading_factor ** self.__iter)
        self.__eval_C()

        self.q_mean = self.q_mean + xi @ (tau * (K @ self.__eps_curr))
        self.Q = (1 - tau) * self.Q + xi @ (tau * (K @ self.C @ K.T + self.P - self.F @ self.P_prev @ self.F.T)) @ xi.T

        IHK = np.eye(self.m) - self.H @ K
        self.r_mean = self.r_mean + tau * (IHK @ self.__eps_curr)
        self.R = (1 - tau) * self.R + tau * (IHK @ (self.C @ IHK.T + self.H @ P_pred @ self.H.T))

    def __eval_C(self):
        self.C = self.C + 1 / self.moving_window * (self.__eps_curr[:, np.newaxis] @ self.__eps_curr[:, np.newaxis].T)
        if self.__iter <= self.moving_window:
            self.__eps_N[self.moving_window - self.__iter] = self.__eps_curr[:]
        else:
            eps = self.__eps_N[self.moving_window - 1]
            self.C = self.C - 1 / self.moving_window * (eps[:, np.newaxis] @ eps[:, np.newaxis].T)
            self.__eps_N = np.roll(self.__eps_N, 1)
            self.__eps_N[0] = self.__eps_curr[:]


