from model import *


def generate_sample(n, m, p, n_obs, u, time_interval, f_name):
    x = np.zeros(shape=(n_obs, n))
    x_noiseless = np.zeros(shape=(n_obs, n))
    y = np.zeros(shape=(n_obs, m))
    mu_state = np.zeros(p)
    mu_obs = np.zeros(m)

    x0 = evaluate_x0()
    t0 = time_interval[0]

    x_noiseless[0] = evaluate_f(x0, t0, u)
    q = evaluate_q(time_interval[0])
    r = evaluate_r(time_interval[1])
    g = evaluate_g(time_interval[0])

    x[0] = x_noiseless[0] + g @ np.random.multivariate_normal(mu_state, q)
    y[0] = evaluate_h(x[0], time_interval[0], u) + np.random.normal(mu_obs, r)

    for j in range(1, n_obs):
        q = evaluate_q(time_interval[j])
        r = evaluate_r(time_interval[j+1])
        g = evaluate_g(time_interval[j])

        x_noiseless[j] = evaluate_f(x_noiseless[j-1], time_interval[j-1], u)
        x[j] = x_noiseless[j] + g @ np.random.multivariate_normal(mu_state, q)
        y[j] = evaluate_h(x[j], time_interval[j], u) + np.random.normal(mu_obs, r)

    with open(f_name + '_x_true.txt', 'w') as file:
        for j in range(0, n_obs):
            for i in range(0, n-1):
                file.write(f'{(x[j,i]):f} ')
            file.write(f'{(x[j, n-1]):f}\n')

    with open(f_name + '_x_true_noiseless.txt', 'w') as file:
        for j in range(0, n_obs):
            for i in range(0, n-1):
                file.write(f'{(x_noiseless[j,i]):f} ')
            file.write(f'{(x_noiseless[j, n-1]):f}\n')

    with open(f_name + '_y.txt', 'w') as file:
        for j in range(0, n_obs):
            for i in range(0, m-1):
                file.write(f'{(y[j, i]):f} ')
            file.write(f'{(y[j, m-1]):f}\n')

    return y


def read_data(size, n_obs, f_name):
    data = np.zeros(shape=(n_obs, size))

    with open(f_name, 'r') as file:
        for j in range(n_obs):
            s = file.readline()
            arr = np.array(s.split(' '))
            data[j] = arr.astype(np.float)
    return data
