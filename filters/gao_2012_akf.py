from filters.discrete_time_kf import DiscreteTimeKF
import numpy as np


class GaoYouKatayamaAKF(DiscreteTimeKF):
    def __init__(self,
                 model,
                 p0,
                 r0,
                 q0,
                 t0,
                 delta_t,
                 q_mean=np.array([0.0]),
                 r_mean=np.array([0.0]),
                 fading_factor=0.7):
        super().__init__(model, p0, r0, q0, t0, delta_t)
        self.__iter = 0
        self.q_mean = q_mean
        self.r_mean = r_mean
        self.x_prev = np.zeros(model.n)
        self.P_prev = np.zeros(shape=(model.n, model.n))
        self.fading_factor = fading_factor

    def __next_iter(self):
        self.__iter += 1

    def predict(self):
        self.__next_iter()
        self.x_prev[:] = self.x
        self.P_prev[:] = self.P
        self.x = self.F @ self.x + self.Psi @ self.u
        self.P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T

    def update(self, y=None):
        if y is None:
            y = self.y

        y_predict = self.H @ self.x
        eps = y - y_predict
        B = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(B)
        self.x = K @ eps + self.x

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
        # self.r_mean = 1 / (self.__iter + 1) * (self.__iter * self.r_mean + eps)
        d = (1 - self.fading_factor) / (1 - self.fading_factor**self.__iter)
        eps_tilda = eps[:, np.newaxis]
        self.R = (1 - d) * self.R + d * (eps_tilda @ eps_tilda.T - self.H @ self.P @ self.H.T)

    def __adapt_Q(self, eps, K):
        gamma = np.linalg.inv(self.G.T @ self.G) @ self.G.T
        d = (1 - self.fading_factor) / (1 - self.fading_factor ** self.__iter)
        # self.q_mean = 1 / self.__iter * ((self.__iter - 1) * self.q_mean + gamma @ (self.x - self.F @ self.x_prev))

        eps_tilda = eps[:, np.newaxis]
        s = gamma @ (K @ eps_tilda @ eps_tilda.T @ K.T + self.P @ self.F @ self.P_prev @ self.F.T) @ gamma.T
        self.Q = (1 - d) * self.Q + d * (gamma @ (K @ eps_tilda @ eps_tilda.T @ K.T + self.P - self.F @ self.P_prev @ self.F.T) @ gamma.T)