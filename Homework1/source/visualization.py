"""
Copyright (C) 2025 Fu Tszkok

:module: visualization
:function: Provides functions to visualize the results of reinforcement learning algorithms.
:author: Fu Tszkok
:date: 2025-10-23
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_convergence_curve(histories_dict, optimal_value):
    """Plots the convergence of the start state's value for multiple algorithms on a single graph.
    :param histories_dict: A dictionary where keys are algorithm names (str) and values are lists of the start state's value at each iteration.
    :param optimal_value: The final, converged optimal value to draw as a horizontal line.
    """
    plt.figure(figsize=(9, 6))

    markers = {'Policy Iteration': 'o-', 'Value Iteration': 'D-', 'Truncated Policy Iteration': 's-'}
    colors = {'Policy Iteration': 'C0', 'Value Iteration': 'C2', 'Truncated Policy Iteration': 'C1'}

    for name, history in histories_dict.items():
        plt.plot(history, markers.get(name, 'x-'), label=name, markersize=4, linewidth=1.5, c=colors.get(name, 'k'))

    # Plot the optimal state value line
    plt.axhline(y=optimal_value, color='red', linestyle='--', label='Optimal State Value (v*)', linewidth=2)

    plt.title('Comparison of Algorithm Convergence')
    plt.xlabel('Iteration (k)')
    plt.ylabel('Value of Start State')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_value_function(env, v, title="State-Value Function (V*)"):
    """Creates a heatmap to visualize the state-value function (V).
    :param env: The maze environment instance.
    :param v: A dictionary mapping states (tuples) to their values.
    :param title: The title for the plot.
    """
    value_grid = np.full(env.maze.shape, np.nan)  # Use NaN for walls
    for state, value in v.items():
        value_grid[state] = value

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(value_grid, cmap='viridis', interpolation='nearest')

    # Add text annotations for the value in each cell
    for r in range(env.height):
        for c in range(env.width):
            if not np.isnan(value_grid[r, c]):
                # Dynamically set text color for better visibility
                val = value_grid[r, c]
                # A simple threshold logic for text color
                text_color = "w" if val < (value_grid[~np.isnan(value_grid)].max() + value_grid[~np.isnan(value_grid)].min()) / 2 else "k"
                ax.text(c, r, f'{val:.2f}', ha='center', va='center', color=text_color)

    plt.colorbar(im, ax=ax, label='State Value')
    ax.set_title(title)
    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    plt.show()


def plot_value_function_snapshots(env, v_history, algorithm_name, num_snapshots=8):
    """Plots several snapshots of the value function's evolution in a grid.
    :param env: The maze environment instance.
    :param v_history: A list of value function dictionaries from the entire iteration.
    :param algorithm_name: The name of the algorithm for the figure's main title.
    :param num_snapshots: The number of snapshots to display (e.g., 6 or 8).
    """
    if len(v_history) < num_snapshots:
        print(f"Warning: History length ({len(v_history)}) is less than requested snapshots ({num_snapshots})."
              f" Displaying all {len(v_history)} steps.")
        num_snapshots = len(v_history)

    # Select evenly spaced indices from the history, including the first and last frames.
    indices = np.linspace(0, len(v_history) - 1, num=num_snapshots, dtype=int)

    # Determine the global color scale from the final value function.
    v_final = v_history[-1]
    final_grid = np.full(env.maze.shape, np.nan)
    for state, value in v_final.items():
        final_grid[state] = value

    vmin = np.nanmin(final_grid)
    vmax = 0.0  # The goal state is the maximum value at 0.0

    # Create the subplot grid.
    # We aim for 2 rows, but adjust if not enough snapshots.
    ncols = (num_snapshots + 1) // 2 if num_snapshots > 1 else 1
    nrows = 2 if num_snapshots > 1 else 1
    # Adjust for odd numbers like 5 or 7, making sure there are enough columns
    if nrows * ncols < num_snapshots:
        ncols += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5), constrained_layout=True)
    fig.suptitle(f'Evolution of Value Function ({algorithm_name})', fontsize=16)

    # Flatten axes array for easy iteration, handling both 1D and 2D cases
    axes = np.array(axes).flatten()

    # Loop through the selected indices and draw each snapshot.
    for i, ax in enumerate(axes):
        if i >= len(indices):
            ax.axis('off')
            continue

        frame_idx = indices[i]
        v = v_history[frame_idx]

        value_grid = np.full(env.maze.shape, np.nan)
        for state, value in v.items():
            value_grid[state] = value

        # Use the global vmin and vmax for consistent coloring
        im = ax.imshow(value_grid, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)

        # Add text annotations for the value in each cell
        for r in range(env.height):
            for c in range(env.width):
                if not np.isnan(value_grid[r, c]):
                    val = value_grid[r, c]
                    # Adjust text color for visibility based on the background color
                    # This logic uses the midpoint of the color scale as a threshold
                    text_color = "w" if val < (vmin + vmax) / 2 else "k"
                    ax.text(c, r, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)

        ax.set_title(f"Iteration: {frame_idx}")
        ax.set_xticks([])  # Hide x-axis ticks for cleaner look
        ax.set_yticks([])  # Hide y-axis ticks

    # Add a single, shared colorbar to the figure.
    fig.colorbar(im, ax=axes.tolist(), orientation='vertical', label='State Value', shrink=0.8)
    plt.show()
