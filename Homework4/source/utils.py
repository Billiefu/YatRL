"""
Copyright (C) 2026 Fu Tszkok

:module: utils
:function: Provides utility classes and functions for the project, including logging (to text and CSV) and plotting results (loss curves, scores, and strategy analysis).
:author: Fu Tszkok
:date: 2026-01-14
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# --- Global Directory Paths ---
# Define and create the primary directories for storing models and results.
RESULT_DIR = "./result"
MODEL_DIR = "./model"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class GameLogger:
    """Handles logging of human-readable game transcripts to a text file."""

    def __init__(self, filename="log.txt"):
        """Initializes the logger and creates the log file with a header.
        :param filename: The name of the file to save the log to. Can include subdirectories.
        """
        self.filepath = os.path.join(RESULT_DIR, filename)
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(self.filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Create the file and write an initial header
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== Liar's Bar Log | {datetime.now()} ===\n")
            f.write(f"File: {filename}\n\n")

    def log(self, text):
        """Appends a line of text to the log file.
        :param text: The string message to be written to the log.
        """
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(str(text) + "\n")


class CSVLogger:
    """Handles logging of structured training data to a CSV file."""

    def __init__(self, filename, fieldnames):
        """Initializes the CSV logger and creates the file with a header row.
        :param filename: The name of the CSV file to save to.
        :param fieldnames: A list of strings representing the column headers.
        """
        self.filepath = os.path.join(RESULT_DIR, filename)
        self.fieldnames = fieldnames

        # Ensure the directory for the CSV file exists
        log_dir = os.path.dirname(self.filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Create the file and write the header
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, data_dict):
        """Writes a new row of data to the CSV file.
        :param data_dict: A dictionary where keys match the `fieldnames` provided during initialization.
        """
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data_dict)


def save_loss_curve(losses, title, filename):
    """Plots the training loss over time and saves it as an image.
    :param losses: A list of loss values.
    :param title: The title for the plot (e.g., the name of the training phase).
    :param filename: The filename for the saved plot image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f"Training Loss - {title}")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()


def plot_scores(scores_history, agent_names, filename="scores_curve.png"):
    """Plots the cumulative score for each agent over all evaluated episodes.
    :param scores_history: A list of lists, where `scores_history[i]` is a list of scores for agent `i`.
    :param agent_names: A list of strings with the names of the agents.
    :param filename: The filename for the saved plot image.
    """
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(agent_names):
        # Calculate the cumulative sum of scores to show the trend over time
        cumsum = np.cumsum(scores_history[i])
        plt.plot(cumsum, label=f"P{i} {name}")

    plt.title("Cumulative Score over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Total Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()


def plot_strategy_stats(doubt_rates, strategy_success_rates, agent_names, filename="strategy_stats.png"):
    """Creates a bar chart comparing the doubt rate and strategy success rate of each agent.
    :param doubt_rates: A list of doubt rate values for each agent.
    :param strategy_success_rates: A list of strategy success rate values for each agent.
    :param agent_names: A list of strings with the names of the agents.
    :param filename: The filename for the saved plot image.
    """
    x = np.arange(len(agent_names))  # The label locations
    width = 0.35  # The width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot bars for doubt rate
    rects1 = ax.bar(x - width / 2, doubt_rates, width, label='Doubt Rate (Aggression)', color='salmon')
    # Plot bars for strategy success rate
    rects2 = ax.bar(x + width / 2, strategy_success_rates, width, label='Strategy Success Rate', color='skyblue')

    # --- Add labels, title, and custom x-axis tick labels ---
    ax.set_ylabel('Rate')
    ax.set_title('Strategy Analysis by Agent Personality')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.legend()

    # Attach a text label above each bar in rects, displaying its height
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()
