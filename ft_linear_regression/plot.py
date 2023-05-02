from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from typing import List, Optional
from numpy import ndarray

from ft_linear_regression.util.calc import calc_y_predict


def plot_line(ax: Axes, x_km: ndarray, theta: ndarray, prev_line: Optional[List[Line2D]] = None) -> List[Line2D]:
    if prev_line:
        prev_line.pop(0).remove()
    return ax.plot(x_km, calc_y_predict(theta, x_km), '-', label="Predictions")
