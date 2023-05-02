from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from typing import List, Optional

from numpy import ndarray


def plot_line(ax: Axes, x_km: ndarray, theta: ndarray, prev_line: Optional[List[Line2D]] = None) -> List[Line2D]:
    if prev_line:
        prev_line.pop(0).remove()
    return ax.plot(x_km, theta[0] + theta[1] * x_km, '-', label="Predictions")
