from filters.discrete_time_kf import DiscreteTimeKF
import numpy as np


class SimpleDiscreteTimeGaoAKF(DiscreteTimeKF):
    def __init__(self,
                 model,
                 p0,
                 r0,
                 q0,
                 t0,
                 delta_t,
                 fading_factor):
        super().__init__(model, p0, r0, q0, t0, delta_t)
        self.__iter = 0
        self.__fading_factor = fading_factor
        self.__eps_curr = np.zeros(self.m)
        self.P_prev = np.zeros(shape=(self.n, self.n))

    def __next_iter(self):
        self.__iter += 1

    def update(self, y=None):
        P_pred = np.zeros(shape=(self.n, self.n))
        if y is None:
            y = np.zeros(self.m)
            y[:] = self.y

        eps = y - self.H @ self.x
        self.__eps_curr[:] = eps

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ eps
        P_pred[:] = self.P
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        self.y = self.H @ self.x
        self.__next_iter()
        self.__adaptR(K, P_pred)
        # self.__adaptQ(K)

        # Обновление матриц модели
        self.t = self.t + self.delta_t
        self.F = self.model.evaluate_f(self.t)
        self.Psi = self.model.evaluate_psi(self.t)
        self.G = self.model.evaluate_g(self.t)
        self.H = self.model.evaluate_h(self.t + self.delta_t)

    def __adaptQ(self, K):
        xi = np.linalg.inv(self.G.T @ self.G) @ self.G.T
        tau = (1 - self.__fading_factor) / (1 - self.__fading_factor ** self.__iter)
        eps = self.__eps_curr[:, np.newaxis] @ self.__eps_curr[:, np.newaxis].T
        self.Q = (1 - tau) * self.Q + xi @ (tau * (K @ eps @ K.T + self.P - self.F @ self.P_prev @ self.F.T)) @ xi.T

    def __adaptR(self, K, P_pred):
        IHK = np.eye(self.m) - self.H @ K
        tau = (1 - self.__fading_factor) / (1 - self.__fading_factor ** self.__iter)
        eps = self.__eps_curr[:, np.newaxis] @ self.__eps_curr[:, np.newaxis].T
        self.R = (1 - tau) * self.R + tau * (IHK @ (eps @ IHK.T + self.H @ P_pred @ self.H.T))

    def reset(self, p0, r0, q0, t0, delta_t):
        self.__init__(self.model, p0, r0, q0, t0, delta_t, self.__fading_factor)

