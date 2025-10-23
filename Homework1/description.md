## HOMEWORK1: Grid Maze Solver

This project is an implementation for the first assignment of the Reinforcement Learning course. The objective is to solve a classic grid maze problem by applying fundamental dynamic programming algorithms, specifically **Policy Iteration** and **Value Iteration**.

### Problem Formulation (MDP)

The grid maze environment is formally modeled as a Markov Decision Process (MDP), as required by the assignment. The key components are defined as follows:

* **States**: The agent's location (coordinates) within the grid.
* **Actions**: A set of four deterministic actions: North (N), East (E), South (S), and West (W).
* **Rewards**: A reward of **-1** is given for each time-step (per move). This reward structure encourages the agent to find the shortest possible path from the start to the goal.
* **Transitions**: The model is deterministic. When the agent takes an action, it moves one step in the corresponding direction unless it is blocked by a wall or the boundary of the maze.

### Implemented Algorithms

This repository provides the code to find the optimal policy (i.e., the shortest path) using the following classical algorithms:

1. **Value Iteration**: An algorithm that iteratively updates the value function for each state until it converges to the optimal value function, from which the optimal policy is then derived.
2. **Policy Iteration**: An algorithm that alternates between two steps:
   * **Policy Evaluation**: Calculating the value function for the current policy.
   * **Policy Improvement**: Greedily improving the policy based on the calculated value function.

### Results

The implementation successfully solves the given maze problem. The results include:

* A **comprehensive report** (over 3 pages) detailing the MDP modeling process, algorithm explanations, and an analysis of the experimental results.
* The final **optimal policy**, which clearly shows the shortest path from the "Start" to the "Goal".
* Visualization of the **optimal value function**, often represented as a heatmap, showing the expected cumulative reward from each state.
