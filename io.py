from model import *


def generate(n, m, p, N, u, time_interval, f_name):
    x = np.zeros(shape=(N, n))
    x_noiseless = np.zeros(shape=(N, n))
    y = np.zeros(shape=(N, m))
    mu = np.zeros(p)

    x0 = evaluate_x0()
    t0 = time_interval[0]

    f = evaluate_f(x0, t0, u)
    q = evaluate_q(time_interval[0])
    r = evaluate_r(time_interval[1])
    g = evaluate_g(time_interval[0])

    x[0] = f + g @ np.random.multivariate_normal(mu, q)
    x_noiseless[0] = f
    mu = np.zeros(m)
    y[0] = evaluate_h(x[0], time_interval[0], u) + np.random.normal(mu, r)

    for j in range(1, N):
        q = evaluate_q(time_interval[j-1])
        r = evaluate_r(time_interval[j])
        g = evaluate_g(time_interval[j-1])

        x_noiseless[j] = evaluate_f(x_noiseless[j-1], time_interval[j-1], u)
        x[j] = evaluate_f(x[j-1], time_interval[j-1], u) + g @ np.random.multivariate_normal(mu, q)
        mu = np.zeros(m)
        y[j] = evaluate_h(x[j], time_interval[j], u) + np.random.normal(mu, r)

    with open(f_name + '_x_true.txt', 'w') as file:
        for j in range(0, N):
            for i in range(0, n-1):
                file.write(f'{(x[j,i]):f} ')
            file.write(f'{(x[j, n-1]):f}\n')

    with open(f_name + '_x_true_noiseless.txt', 'w') as file:
        for j in range(0, N):
            for i in range(0, n-1):
                file.write(f'{(x_noiseless[j,i]):f} ')
            file.write(f'{(x_noiseless[j, n-1]):f}\n')

    with open(f_name + '_y.txt', 'w') as file:
        for j in range(0, N):
            for i in range(0, m-1):
                file.write(f'{(y[j, i]):f} ')
            file.write(f'{(y[j, m-1]):f}\n')

    return y
