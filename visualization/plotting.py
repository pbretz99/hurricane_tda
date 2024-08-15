import matplotlib.pyplot as plt
import numpy as np


def add_circ(ax: plt.Axes, x: float, y: float, r: float, **kwargs) -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    x_vals = x + r * np.cos(theta)
    y_vals = y + r * np.sin(theta)
    ax.plot(x_vals, y_vals, **kwargs)

def add_square(ax: plt.Axes, x: float, y: float, r: float, **kwargs) -> None:
    x_vals = [x - r, x + r, x + r, x - r, x - r]
    y_vals = [y - r, y - r, y + r, y + r, y - r]
    ax.plot(x_vals, y_vals, **kwargs)
