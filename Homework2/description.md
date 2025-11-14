## HOMEWORK2: Cliff Walk with TD Learning

This project is an implementation for a reinforcement learning course assignment. The objective is to solve the classic **Cliff Walk** problem by applying and comparing fundamental **Temporal Difference (TD) learning algorithms**, specifically **SARSA** and **Q-learning**, along with their variants.

### Problem Formulation (MDP)

The Cliff Walk environment is formally modeled as a Markov Decision Process (MDP). The key components are defined as follows:

* **States**: The agent's location (coordinates) within the $4\times12$ grid.
* **Actions**: A set of four deterministic actions: North (N), East (E), South (S), and West (W).
* **Rewards**: A reward of **-1** is given for each step taken. A large negative reward of **-100** is given for stepping into a "cliff" tile. This reward structure encourages the agent to find the shortest path while strictly avoiding the cliff.
* **Transitions**: The model is deterministic. When the agent takes an action, it moves one step in the corresponding direction. However, if the move leads into a cliff tile, the agent is immediately sent back to the start position for that episode.

### Implemented Algorithms

This repository provides the code to find and compare policies using the following TD learning algorithms:

1. **SARSA (State-Action-Reward-State-Action)**: An **on-policy** TD algorithm that learns the value of a policy while actively following it, thereby accounting for the risks associated with its own exploration strategy.
2. **Q-learning**: An **off-policy** TD algorithm that directly learns the optimal policy's value function, regardless of the exploration strategy being used during training.
3. **Expected SARSA**: An on-policy variant of SARSA that reduces learning variance by calculating the expected value over all possible next actions, often leading to more stable performance.
4. **N-step SARSA**: A more general on-policy algorithm that looks ahead $n$ steps to update Q-values, bridging the gap between single-step TD methods and Monte Carlo methods.

### Results

The implementation successfully solves the Cliff Walk problem and provides a clear comparison between on-policy and off-policy control methods. The results include:

* A **comprehensive report** (over 3 pages) detailing the Cliff Walk MDP formulation, algorithm implementations, and a comparative analysis of the experimental results.
* The final **learned policies** from each algorithm, highlighting the classic trade-off between the "safe" path learned by SARSA and the "optimal" (but risky) path learned by Q-learning.
* Visualizations including **learning curves** (rewards per episode) to compare the performance, stability, and convergence speed of the different algorithms, as well as **value function heatmaps** for the final learned policies.
