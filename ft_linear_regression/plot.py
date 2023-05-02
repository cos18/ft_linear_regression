from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from typing import List, Optional
from ft_linear_regression.util.calc import calc_predict


def plot_data(ax: Axes, data: List[List[float]]):
    ax.plot([d[0] for d in data], [d[1] for d in data], 'o', color='tab:brown')


def plot_line(ax: Axes, data: List[List[float]], theta: List[float], prev_line: Optional[List[Line2D]] = None) -> List[Line2D]:
    if prev_line:
        prev_line.pop(0).remove()
    return ax.plot([d[0] for d in data], [calc_predict(theta, d[0]) for d in data], '-')
