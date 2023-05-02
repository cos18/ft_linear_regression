from typing import Tuple
import numpy as np
from numpy import ndarray


def calc_predict(model_info: ndarray, mileage: float) -> float:
    return model_info[0] + model_info[1] * (mileage - model_info[2]) / model_info[3]


def calc_array_norm(model_info: ndarray, x_km: ndarray):
    return (x_km - model_info[2]) / model_info[3]


def calc_y_predict(model_info: ndarray, x_km: ndarray):
    x_norm = calc_array_norm(model_info, x_km)
    return model_info[0] + model_info[1] * x_norm


def calc_error_array(model_info: ndarray, x_km: ndarray, y_price: ndarray) -> ndarray:
    return calc_y_predict(model_info, x_km) - y_price


def calc_gradient(model_info: ndarray, x_km: ndarray, y_price: ndarray) -> ndarray:
    x_norm = calc_array_norm(model_info, x_km)
    y_error = calc_error_array(model_info, x_km, y_price)
    return np.array([y_error.mean(), (y_error * x_norm).mean()])


def calc_mse(model_info: ndarray, x_km: ndarray, y_price: ndarray) -> float:
    return (calc_error_array(model_info, x_km, y_price)**2).mean()
