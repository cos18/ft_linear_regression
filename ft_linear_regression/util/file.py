from typing import Tuple
import numpy as np
from numpy import ndarray


def get_data() -> Tuple[ndarray, ndarray]:
    try:
        with open('ft_linear_regression/data/data.csv', 'r') as data_file:
            data_file.readline()

            data_list = []
            data_org = data_file.readline()

            try:
                while data_org != '':
                    data_list.append(list(map(float, data_org.split(','))))
                    data_org = data_file.readline()
            except ValueError:
                raise ValueError('Error while parsing data!')
            result = np.array(data_list).transpose()
            return result[0], result[1]
    except OSError:
        raise OSError('Error when opening data file!')


def get_model_info() -> ndarray:
    """
    get model info from `data/model_info`
    :return: np.array [theta0, theta1, x_mean, x_std]
    """
    try:
        with open('ft_linear_regression/data/model_info', 'r') as theta_file:
            theta_org = theta_file.readline().split(',')
            try:
                return np.array(list(map(float, theta_org)))
            except ValueError:
                raise ValueError('Error while parsing model info!')
    except OSError:
        raise OSError('Error when opening model info file!')


def set_model_info(model_info: ndarray):
    try:
        with open('ft_linear_regression/data/model_info', 'w') as theta_file:
            theta_file.write(','.join(np.char.mod('%f', model_info)))
    except OSError:
        raise OSError('Error when writing model info file!')


def reset_model_info():
    set_model_info(np.concatenate([np.zeros(3), [1]], axis=None))
