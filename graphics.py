import matplotlib.pyplot as plt
import numpy as np


def visualize_filter_results(n, label, suptitle, x_real_data, x_data, time):
    plt.Figure()
    for i in range(n):
        plt.title(label)
        plt.suptitle(suptitle)
        plt.subplot(n, 1, i + 1)
        plt.plot(time, x_data[:, i], label='фильтр')
        plt.plot(time, x_real_data[:, i], label='истина')
        plt.legend()
    plt.show()


def visualize_covariance(n, label, covar,  time):
    plt.Figure()

    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(time, covar[i], label='P'+str(i))
        plt.legend()
    plt.show()


def draw_states(label, states, time):
    plt.Figure()
    plt.suptitle(label)
    styles = ['.--r', 'b', '*-.g', '-c']
    i = 0

    for data, name in states:
        for j in range(np.shape(data)[1]):
            plt.subplot(2, 1, j + 1)
            plt.plot(time, data[:, j], styles[i], label=name, linewidth=0.5, mew=0.5)
        i += 1
    plt.legend()
    plt.show()
