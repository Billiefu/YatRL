"""
Copyright (C) 2025 Fu Tszkok

:module: maze
:function: Defines a customizable grid-world maze environment for reinforcement learning.
:author: Fu Tszkok
:date: 2025-10-20
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np

from layout import maze3


class MazeEnvironment:
    """A grid-world maze environment for reinforcement learning."""

    def __init__(self, maze=None, reward_step=-1.0, reward_goal=0.0):
        """Initializes the maze environment.
        :param maze: A 2D NumPy array or list of lists defining the maze layout.
                     (0: path, 1: wall, 2: start, 3: goal). If None, a default maze is used.
        :param reward_step: The reward received for each step taken that does not reach the goal.
        :param reward_goal: The reward received for reaching the goal state.
        """
        # If no custom maze is provided, use a default layout.
        if maze is None:
            self.maze = maze3
        else:
            self.maze = np.array(maze)

        # Validate that the maze contains exactly one start and one goal point.
        if 2 not in self.maze or 3 not in self.maze:
            raise ValueError("The maze must include a starting point (2) and an endpoint (3).")
        if np.argwhere(self.maze == 2).shape[0] > 1 or np.argwhere(self.maze == 3).shape[0] > 1:
            raise ValueError("The maze can only contain one start point (2) and one end point (3).")

        # Define core environment properties based on the maze layout.
        self.height, self.width = self.maze.shape
        self.start_pos = tuple(np.argwhere(self.maze == 2)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 3)[0])

        # Define the action space as a dictionary mapping action names to coordinate changes.
        self.actions = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
        self.action_keys = list(self.actions.keys())

        # Define the state space as a list of all non-wall coordinates.
        self.states = []
        for r in range(self.height):
            for c in range(self.width):
                if self.maze[r, c] != 1:
                    self.states.append((r, c))

        # Store the reward structure.
        self.reward_step = reward_step
        self.reward_goal = reward_goal

    def is_valid_pos(self, pos):
        """Checks if a given position is within the maze boundaries and is not a wall.
        :param pos: A tuple (row, col) representing the position to check.
        :return: True if the position is valid, False otherwise.
        """
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False  # Position is outside the grid.
        elif self.maze[r, c] == 1:
            return False  # Position is a wall.
        return True

    def step(self, current_pos, action):
        """Executes an action from a given state and returns the outcome.
        :param current_pos: The agent's current state as a tuple (row, col).
        :param action: The action to be taken (e.g., 'N', 'S', 'W', 'E').
        :return: A tuple (next_pos, reward, done) where:
                 - next_pos is the agent's new state.
                 - reward is the immediate reward received.
                 - done is a boolean indicating if the goal has been reached.
        """
        # If the agent is already at the goal, it stays there with zero reward.
        if current_pos == self.goal_pos:
            return self.goal_pos, 0, True

        move = self.actions.get(action)
        if move is None:
            raise ValueError(f"Invalid action: {action}")

        # Calculate the potential next position based on the action.
        next_pos_theoretical = (current_pos[0] + move[0], current_pos[1] + move[1])

        # Determine the actual next position based on maze rules (walls, boundaries).
        if self.is_valid_pos(next_pos_theoretical):
            next_pos = next_pos_theoretical
        else:
            # If the move is invalid, the agent stays in the same place.
            next_pos = current_pos

        # Check if the new position is the goal state.
        done = (next_pos == self.goal_pos)
        # Assign reward based on whether the goal was reached.
        reward = self.reward_goal if done else self.reward_step

        return next_pos, reward, done

    def render(self):
        """Prints a visual representation of the maze layout to the console."""
        symbols = {0: '⬜', 1: '⬛', 2: '✳️', 3: '✅'}
        for r in range(self.height):
            for c in range(self.width):
                print(symbols[self.maze[r, c]], end="")
            print()
        print()

    def print_policy(self, policy):
        """Prints a visual representation of a policy on the maze grid.
        :param policy: A dictionary mapping states (tuples) to actions (strings).
        """
        policy_symbols = {'N': '🔼', 'S': '🔽', 'W': '◀️', 'E': '▶️'}
        # Use dtype=object to prevent NumPy from truncating emoji characters.
        policy_grid = np.full(self.maze.shape, ' ', dtype=object)

        # Populate the grid with symbols for the calculated policy.
        for state, action in policy.items():
            if state == self.goal_pos:
                policy_grid[state] = '✅'
            else:
                policy_grid[state] = policy_symbols[action]

        # Overlay symbols for walls and the start position for clarity.
        for r in range(self.height):
            for c in range(self.width):
                if self.maze[r, c] == 1:
                    policy_grid[r, c] = '⬛'
                elif self.maze[r, c] == 2 and (r, c) not in policy:
                    policy_grid[r, c] = '✳️'

        # Print the final policy grid row by row.
        for r in range(self.height):
            for c in range(self.width):
                print(policy_grid[r, c], end="")
            print()
        print()
