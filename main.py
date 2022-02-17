# TODO 1) Генерация данных по модели состояний/измерений
# TODO 2) Сравнить с дискретным фильтром Калмана
# TODO 3) Визуализировать результаты


def graphics(x_true, x_dckf, x_kalman, time_interval):
    plt.Figure()
    plt.plot(time_interval, x_true[:, 0])
    plt.plot(time_interval, x_dckf[:, 0])
    plt.plot(time_interval, x_kalman[:, 0])
    plt.legend(['x true', 'x dckf', 'x kalman'])
    plt.show()


from discrete_time_ckf import *
from in_out import *
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

n = 2
m = 1
p = 2
t = 1
u = np.array([])
n_obs = 100
time_interval = np.zeros(shape=(n_obs+1, 1))

for j in range(0, n_obs + 1):
    time_interval[j] = j * t

p0 = evaluate_p0()
x0 = evaluate_x0()
r0 = evaluate_r()
q0 = evaluate_q()

generate_sample(n, m, p, n_obs, u, time_interval, 'simple')
y = read_data(m, n_obs, 'simple_y.txt')
x_true = read_data(n, n_obs, 'simple_x_true.txt')
x_true_noiseless = read_data(n, n_obs, 'simple_x_true_noiseless.txt')
x_dckf = np.zeros(shape=(n_obs, n))
x_kf = np.zeros(shape=(n_obs, n))

f_dckf = DiscreteTimeCKF(n, m, p0, r0, q0, x0)
f_kf = KalmanFilter(dim_x=n, dim_z=m)
f_kf.x = np.array([0, 0])
f_kf.F = np.array([[1, 1], [0, 1]])
f_kf.H = np.array([[1., 0]])
f_kf.P = np.array([[1, 0], [0, 1]])
f_kf.R = np.array([[1.]])
f_kf.Q = np.array([[1, 0], [0, 1]])


for i in range(0, n_obs):
    f_kf.predict()
    f_dckf.predict()
    f_kf.update(y[i])
    f_dckf.update(y[i])
    x_kf[i] = f_kf.x
    x_dckf[i] = f_dckf.x

print('x_dckf: ', x_dckf)
print('x_kf', x_kf)
graphics(x_true, x_kf, x_dckf, time_interval[1:])
