import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
# This module generates figures describing the environment and path returned by A*


# Plot grid environment with A* path, traveled path, start and goal cells.
def plot_grid(grid, start = None, goal = None, history = None, fig = None, ax = None, brightness = 0.5):
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    norm = colors.Normalize(vmin=-0, vmax=1)
    ax.imshow(brightness*grid, cmap='gray_r', norm=norm)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, grid.shape[0], 1));
    ax.set_yticks(np.arange(-0.5, grid.shape[1], 1));
    
    if start is not None and goal is not None:
        current = goal
        while current is not None and current != start:
            next = current
            current = current.parent
            if current is not None:
                plt.plot(current.x, current.y, marker='o', markersize=3, color="b")
                plt.arrow(current.x, current.y, 0.7*(next.x - current.x), 0.7*(next.y - current.y), head_width=0.2, head_length=0.2, fc="b", ec="b", linestyle=':')
        ax.plot(goal.x, goal.y, marker='o', markersize=3, color="r")
        ax.plot(start.x, start.y, marker='o', markersize=3, color="r")

    if history is not None:
        for i in range(len(history)):
            current = history[i]
            plt.plot(current.x, current.y, marker='o', markersize=3, color="xkcd:orange")
            if (i < len(history) - 1):
                next = history[i+1]
                plt.arrow(current.x, current.y, 0.7 * (next.x - current.x), 0.7 * (next.y - current.y), head_width=0.2,
                          head_length=0.2, fc="xkcd:orange", ec="xkcd:orange")
        current = history[-1]
        # Plot robot
        ax.plot(current.x, current.y, marker='^', markersize=9, color="g")


# Generate grid based on the current graph in Robot class. This is to visualize the current map estimation.
def graph2grid(grid, graph, start, end):
    new_grid = np.zeros(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            new_grid[i, j] = graph[j][i].blocked_prob
    current = end
    while current != start and current is not None:
        current = current.parent
    return new_grid
