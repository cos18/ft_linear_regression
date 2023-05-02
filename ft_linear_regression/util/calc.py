import numpy as np
from numpy import ndarray


def calc_predict(theta: ndarray, mileage: float) -> float:
    return theta[0] + theta[1] * mileage


def calc_error_array(theta: ndarray, x_km: ndarray, y_price: ndarray) -> ndarray:
    y_predict = theta[0] + theta[1] * x_km
    return y_predict - y_price


def calc_gradient(theta: ndarray, x_km: ndarray, y_price: ndarray) -> ndarray:
    y_error = calc_error_array(theta, x_km, y_price)
    return np.array([y_error.mean(), (y_error * x_km).mean()])


def calc_mse(theta: ndarray, x_km: ndarray, y_price: ndarray) -> float:
    return (calc_error_array(theta, x_km, y_price)**2).mean()
