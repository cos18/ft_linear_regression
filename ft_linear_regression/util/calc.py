from typing import List


def calc_predict(theta: List[float], mileage: float) -> float:
    return theta[0] + theta[1] * mileage


def calc_gradient(theta: List[float], data: List[List[float]], lr: float) -> List[float]:
    result = [0, 0]
    for d in data:
        diff = (calc_predict(theta, d[0]) - d[1]) * lr / len(data)
        result[0] += diff
        result[1] += (diff * d[0])
    return result


def calc_mse(theta: List[float], data: List[List[float]]) -> float:
    return sum([(d[1] - calc_predict(theta, d[0])) ** 2 / len(data) for d in data])
