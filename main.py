from input_output import generate_sample, read_data
from graphics import draw_states
import numpy as np
import pandas as pd

modelNum = '1'
match modelNum:
    case '1':
        import models.system_position_control
        model = models.system_position_control.SystemPositionControl()
    case '2':
        import models.linear_uniformly_accelerated_motion
        model = models.linear_uniformly_accelerated_motion.LinUniAccMotion()
    case '3':
        import models.gyroscope_model
        model = models.gyroscope_model.GyroscopeModel()
    case '4':
        print('Поддержка своей модели отсутствует')
        exit(0)
    case _:
        print('Некорректное значение')
        exit(0)

n, m, t0, t_max, delta_t, n_obs = model.n, model.m, 0, 10., 0.1, 100

R0_true = model.evaluate_r(t0 + delta_t)
R_true = np.zeros(shape=(n_obs + 1, m, m))
Q0_true = model.evaluate_q(t0)
Q_true = np.zeros(shape=(n_obs + 1, model.p, model.p))

time = np.linspace(t0, t_max, n_obs + 1)
armse_x = np.zeros(n)
armse_y = np.zeros(m)
x_filtered = np.zeros(shape=(n_obs + 1, n))
x_filtered[0] = model.evaluate_x0()
y_filtered = np.zeros(shape=(n_obs, m))
r_estimated = np.zeros(shape=(n_obs + 1, m, m))
q_estimated = np.zeros(shape=(n_obs + 1, model.p, model.p))


R0_list = [(0.1 * R0_true, 'R small'), (R0_true, 'R true'), (10 * R0_true, 'R big')]
# R0_list = [(R0_true, 'R true')]
Q0_list = [(0.1 * Q0_true, 'Q small'), (Q0_true, 'Q true'), (10 * Q0_true, 'Q big')]
# Q0_list = [(100 * Q0_true, 'Q big')]
# Q0_list = [(Q0_true, 'Q true')]
b_list = np.linspace(0.01, 0.99, 3)
# b_list = [1]

columns = list(['R0', 'Q0', 'b'])
for i in range(n):
    columns.append('rmse(x' + str(i+1) + ')')
columns.append('norm of rmse(x)')
for i in range(m):
    columns.append('rmse(y' + str(i+1) + ')')
columns.append('norm of rmse(y)')

b_research_table = pd.DataFrame(columns=columns)

for i in range(n_obs + 1):
    R_true[i] = model.evaluate_r(t0 + delta_t * i)
    Q_true[i] = model.evaluate_q(t0 + delta_t * i)

filter_name = None

fname = 'b_research'
launches = 100
states = []

# for i in range(launches):
#     fname = 'b_research' + str(i)
#     y = generate_sample(model, n_obs, time, fname)

for R0, R_name in R0_list:
    for Q0, Q_name in Q0_list:
        r_estimated[0] = R0
        q_estimated[0] = Q0

        for b in b_list:
            states = []
            print(f'\n\nb = {b} R0 = {R0[0,0]} Q0 = {Q0[0,0]}')
            # P = np.diag(np.ones(n) * 1e-2)
            P = np.diag(np.ones(n) * 100)

            # Выбор фильтра
            # print('Выберите алгоритм фильтрации:')
            # print('1. Фильтр Калмана')
            # print('2. Адаптивная модификация Sage и Husa')
            # print('3. Адаптивная модификация Gao, You и Katayama')
            # print('4. Адаптивная модификация Гао(упрощенная)')
            # print('5. Адаптивная модификация Юань-Леи')

            # filterNum = input('Номер:')
            for filterNum in ['1', '2']:
                rmse_x = np.zeros(n_obs)
                armse_x.fill(0.0)
                armse_y.fill(0.0)
                # filterNum = '4'
                isEqualDimensions = False
                match filterNum:
                    case '1':
                        import filters.discrete_time_kf as kf
                        kalman_filter = kf.DiscreteTimeKF(model,
                                                          P,
                                                          R0,
                                                          Q0,
                                                          t0,
                                                          delta_t)
                        filter_name = 'фильтр Калмана'
                    case '2':
                        import filters.sage_husa_akf as kf

                        kalman_filter = kf.SageHusaAKF(model,
                                                       P,
                                                       R0,
                                                       Q0,
                                                       t0,
                                                       delta_t)
                        filter_name = 'Адаптивная модификация Sage и Husa(1969)'
                    case '3':
                        import filters.gao_2012_akf as kf

                        kalman_filter = kf.GaoYouKatayamaAKF(model,
                                                             P,
                                                             R0,
                                                             Q0,
                                                             t0,
                                                             delta_t,
                                                             fading_factor=b)
                        filter_name = 'Адаптивная модификация Gao, You и Kayatama(2012)'
                    case '4':
                        import filters.gao_akf_discrete_simple
                        kalman_filter = filters.gao_akf_discrete_simple.SimpleDiscreteTimeGaoAKF(model,
                                                                                                 P,
                                                                                                 R0,
                                                                                                 Q0,
                                                                                                 t0,
                                                                                                 delta_t,
                                                                                                 b)
                        filter_name = 'Гао(упрощенная)'
                    case '5':
                        print('Not supported')
                        exit(1)

                # fname = input('Введите название для файла с данными:')
                # launches = int(input('Введите число запусков системы:'))

                for i in range(launches):
                    fname = 'b_research' + str(launches)
                    x_true = read_data(n, n_obs, 'data/' + fname + '_x_true.txt')
                    x_true_noiseless = read_data(n, n_obs, 'data/' + fname + '_x_true_noiseless.txt')
                    y = read_data(m, n_obs, 'data/' + fname + '_y.txt')

                    for j in range(n_obs):
                        kalman_filter.predict()
                        kalman_filter.update(y[j])
                        armse_x = armse_x + np.square(x_true_noiseless[j] - kalman_filter.x)
                        armse_y = armse_y + np.square(y[j] - kalman_filter.y)

                        for k in range(n):
                            rmse_x[j] += x_true_noiseless[j, k] - kalman_filter.x[k]
                        rmse_x[j] = np.sqrt(rmse_x[j])

                        x_filtered[j + 1] = kalman_filter.x.copy()
                        y_filtered[j] = kalman_filter.y.copy()
                        r_estimated[j + 1] = np.diag(kalman_filter.R)
                        q_estimated[j + 1] = kalman_filter.Q.copy()
                    kalman_filter.reset(P, R0, Q0, t0, delta_t)

                armse_x = np.sqrt(armse_x / (launches * n_obs))
                armse_y = np.sqrt(armse_y / (launches * n_obs))
                # for i in range(n):
                #     print(f'RMSE(x{i+1}): {armse_x[i]}')
                # print(f'Норма RMSE(x): {np.linalg.norm(armse_x)}')
                # for j in range(m):
                #     print(f'RMSE(y{j+1}): {armse_y[j]}')

                new_row = np.array([R_name, Q_name, b], dtype=object)
                for i in range(n):
                    new_row = np.append(new_row, armse_x[i])
                new_row = np.append(new_row, np.linalg.norm(armse_x))
                for i in range(m):
                    new_row = np.append(new_row, armse_y[i])
                new_row = np.append(new_row, np.linalg.norm(armse_y))
                b_research_table = pd.concat([b_research_table, pd.DataFrame([new_row], columns=columns)])

                for i in range(n):
                    print(f'{armse_x[i]}', end="\t")
                print(f'{np.linalg.norm(armse_x)}', end="\t")
                for j in range(m):
                    print(f'{armse_y[j]}', end="\t")
                print()
                states.append((x_filtered.copy(), filter_name))

            states.append((np.concatenate((model.evaluate_x0()[np.newaxis, :], x_true), axis=0), 'Зашумленный X'))
            states.append((np.concatenate((model.evaluate_x0()[np.newaxis, :], x_true_noiseless), axis=0), 'Незашумленный X'))
            draw_states(f'{R_name} {Q_name}', states, time)
# b_research_table.to_excel('sage_KF_hashlamon.xlsx')
