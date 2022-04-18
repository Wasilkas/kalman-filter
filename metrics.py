import numpy as np

def RMSE(n_obs, real_data, calculated_data):
    error = np.zeros(shape=(np.shape(real_data)[1], 1))
    for i in range(n_obs):
        error = error + np.square(real_data[i] - calculated_data[i])[:, np.newaxis]
    return np.sqrt(error)
