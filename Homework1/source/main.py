"""
Copyright (C) 2025 Fu Tszkok

:module: main
:function: Executes and demonstrates reinforcement learning algorithms on a maze environment.
:author: Fu Tszkok
:date: 2025-10-23
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

from iteration import *
from layout import *
from maze import MazeEnvironment
from visualization import *


# --- 1. Environment Setup ---

# Initialize a new instance of the MazeEnvironment using its default layout.
maze_env = MazeEnvironment()
# maze_env = MazeEnvironment(maze=maze1)
# maze_env = MazeEnvironment(maze=maze2)
# maze_env = MazeEnvironment(maze=maze4)
print("------- Maze Layout -------")
# Render the maze structure to the console for visual inspection.
maze_env.render()

# Dictionaries to store results for combined plot
histories_for_plot = {}
final_v_values = {}


# --- 2. Value Iteration ---

# Run the Value Iteration algorithm to find the optimal value function and policy.
print("--- Value Iteration ---")
V_vi, policy_vi, v_history_vi = value_iteration(maze_env)

# Extract start state value history for the convergence plot
final_v_values['Value Iteration'] = V_vi
histories_for_plot['Value Iteration'] = [v[maze_env.start_pos] for v in v_history_vi]

# Display the resulting policy visually on the maze grid.
print("Optimal Policy Found:")
maze_env.print_policy(policy_vi)

# Display the evolution of Value Iteration as static snapshots
plot_value_function_snapshots(maze_env, v_history_vi, "Value Iteration", num_snapshots=6)


# --- 3. Policy Iteration ---

# Run the Policy Iteration algorithm, which alternates between policy evaluation and policy improvement until convergence.
print("--- Policy Iteration ---")
V_pi, policy_pi, v_history_pi = policy_iteration(maze_env)

# Extract start state value history for the convergence plot
final_v_values['Policy Iteration'] = V_pi
histories_for_plot['Policy Iteration'] = [v[maze_env.start_pos] for v in v_history_pi]

# Display the resulting policy visually on the maze grid.
print("Optimal Policy Found:")
maze_env.print_policy(policy_pi)

# Display the evolution of Value Iteration as static snapshots
plot_value_function_snapshots(maze_env, v_history_pi, "Policy Iteration", num_snapshots=6)


# --- 4. Truncated Policy Iteration ---

# Run the Policy Iteration algorithm, which alternates between policy evaluation and policy improvement until convergence.
print("\n--- Truncated Policy Iteration ---")
V_tpi, policy_tpi, v_history_tpi = truncated_policy_iteration(maze_env)

# Extract start state value history for the convergence plot
final_v_values['Truncated Policy Iteration'] = V_tpi
histories_for_plot['Truncated Policy Iteration'] = [v[maze_env.start_pos] for v in v_history_tpi]

# Display the resulting policy visually on the maze grid.
print("Optimal Policy Found:")
maze_env.print_policy(policy_tpi)

# Display the evolution of Value Iteration as static snapshots
plot_value_function_snapshots(maze_env, v_history_tpi, "Truncated Policy Iteration", num_snapshots=6)


# --- 5. Final Visualizations ---

# Show the final V-function heatmap for one of the algorithms (they should be identical)
visualize_value_function(maze_env, V_vi, "Final Optimal Value Function (V*)")

# Show the combined convergence plot
# We use the final start state value from value iteration as the v*
optimal_start_value = V_vi[maze_env.start_pos]
plot_convergence_curve(histories_for_plot, optimal_start_value)
