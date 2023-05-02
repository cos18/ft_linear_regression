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


def get_theta() -> ndarray:
    try:
        with open('ft_linear_regression/data/theta', 'r') as theta_file:
            theta_org = theta_file.readline().split(',')
            try:
                return np.array(list(map(float, theta_org)))
            except ValueError:
                raise ValueError('Error while parsing theta!')
    except OSError:
        raise OSError('Error when opening theta file!')


def set_theta(theta: ndarray):
    try:
        with open('ft_linear_regression/data/theta', 'w') as theta_file:
            theta_file.write(','.join(np.char.mod('%f', theta)))
    except OSError:
        raise OSError('Error when writing theta file!')


def reset_theta():
    set_theta(np.zeros(2))
