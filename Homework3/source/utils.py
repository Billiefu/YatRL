"""
Copyright (C) 2025 Fu Tszkok

:module: utils
:function: Provides utility functions for visualizing training progress (loss, win rate) and game states.
:author: Fu Tszkok
:date: 2025-11-05
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def plot_loss(save_dir, loss_data):
    """Plots the training loss curve and saves it to a file.
    :param save_dir: The directory where the plot image will be saved.
    :param loss_data: A list of tuples (batch_number, loss_value).
    """
    if not loss_data:
        return

    plt.figure(figsize=(10, 6))
    # Unpack batch numbers and loss values
    batches = [x[0] for x in loss_data]
    losses = [x[1] for x in loss_data]

    # Plot the loss curve
    plt.plot(batches, losses, label='Training Loss', color='b', linewidth=1.5)
    plt.title('AlphaZero Training Loss')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Save the figure to the specified directory
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path)
    plt.close()


def plot_win_rate(save_dir, win_rate_data):
    """Plots the win rate evaluation curve and saves it to a file.
    :param save_dir: The directory where the plot image will be saved.
    :param win_rate_data: A list of tuples (batch_number, win_rate).
    """
    if not win_rate_data:
        return

    plt.figure(figsize=(10, 6))
    batches = [x[0] for x in win_rate_data]
    rates = [x[1] for x in win_rate_data]

    # Plot the win rate curve with markers
    plt.plot(batches, rates, label='Win Rate vs Pure MCTS', color='r', linewidth=1.5, marker='o')

    plt.title('AlphaZero Win Rate Evaluation')
    plt.xlabel('Batch Number')
    plt.ylabel('Win Rate')
    # Set y-axis limits to clearly show 0 to 1 range
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    save_path = os.path.join(save_dir, 'win_rate_curve.png')
    plt.savefig(save_path)
    plt.close()


def plot_games(save_dir, board_states_list, batch_idx, board_width, board_height):
    """Visualizes the final board states of evaluation games.
    :param save_dir: The directory where the image will be saved.
    :param board_states_list: A list of dictionaries representing board states (move -> player).
    :param batch_idx: The current training batch index (used for naming the file).
    :param board_width: The width of the board.
    :param board_height: The height of the board.
    """
    num_games = len(board_states_list)
    if num_games == 0:
        return

    # Determine grid layout for subplots (5 columns)
    cols = 5
    rows = (num_games + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    # Flatten axes array for easy iteration if multiple subplots exist
    axes = axes.flatten() if num_games > 1 else [axes]

    for i, states in enumerate(board_states_list):
        ax = axes[i]
        ax.set_title(f"Game {i + 1}")
        ax.set_aspect('equal')

        # Set axis limits to center the grid
        ax.set_xlim(-0.5, board_width - 0.5)
        ax.set_ylim(-0.5, board_height - 0.5)

        # Draw grid lines
        for x in range(board_width):
            ax.plot([x, x], [0, board_height - 1], color='gray', linewidth=1, zorder=0)
        for y in range(board_height):
            ax.plot([0, board_width - 1], [y, y], color='gray', linewidth=1, zorder=0)

        ax.set_xticks([])
        ax.set_yticks([])
        # Invert y-axis to match matrix coordinates (top-left is 0,0)
        ax.invert_yaxis()

        move_history = list(states.keys())

        # Draw stones for each move
        for step_idx, move in enumerate(move_history):
            player = states[move]
            h = move // board_width
            w = move % board_width

            # Define stone appearance based on player ID
            if player == 1:
                face_color = 'black'
                edge_color = 'black'
                text_color = 'white'
            else:
                face_color = 'white'
                edge_color = 'black'
                text_color = 'black'

            # Draw the stone as a circle
            circle = patches.Circle((w, h), radius=0.4, facecolor=face_color, edgecolor=edge_color, linewidth=1.5, zorder=10)
            ax.add_patch(circle)

            # Label the stone with the move number
            ax.text(w, h, str(step_idx + 1), horizontalalignment='center', verticalalignment='center', fontsize=10, color=text_color, weight='bold', zorder=11)

    # Hide unused subplots
    for j in range(num_games, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'eval_batch_{batch_idx}.png')
    plt.savefig(save_path)
    plt.close()
