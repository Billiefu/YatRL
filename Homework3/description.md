## HOMEWORK3: Gomoku with AlphaZero

This project is an implementation for a reinforcement learning course assignment. The objective we chose is to design an intelligent agent to play the board game **Gomoku** (Five-in-a-Row) by replicating the **AlphaZero** algorithm, which combines **Monte Carlo Tree Search (MCTS)** with **Deep Reinforcement Learning (DRL)** via self-play mechanisms.

### Problem Formulation (MDP)

The Gomoku environment is formally modeled as a two-player, zero-sum, perfect-information Markov Decision Process (MDP). The key components are defined as follows:

* **States**: The game state is represented as a $4\times8\times8$ feature tensor. This includes planes for the current player's stones, the opponent's stones, the location of the last move, and a plane indicating the current player's color.
* **Actions**: The action space is discrete, consisting of placing a stone on any empty intersection of the $8\times8$ grid (up to 64 possible actions).
* **Rewards**: A sparse reward signal is used. A reward of +1 is given for winning (forming a continuous line of 5 stones), -1 for losing, and 0 for a draw or any intermediate non-terminal state.
* **Transitions**: The model is deterministic. When an agent executes an action, the board configuration updates, and the turn switches to the opponent. The game terminates when a player wins or the board is full.

### Implemented Algorithms

This repository provides the code to train a superhuman agent from scratch using the following core components of the AlphaZero framework:

1. **Monte Carlo Tree Search (MCTS)**: A heuristic search algorithm used for decision-making. It employs the **PUCT** (Predictor + Upper Confidence Bound applied to Trees) formula to balance exploration and exploitation, serving as a powerful policy improvement operator during self-play.
2. **Policy-Value Network**: A deep **Residual Network (ResNet)** that takes the raw board state as input and outputs both a probability distribution over valid moves (Policy Head) and a scalar evaluation of the current position (Value Head), guiding the MCTS simulations.
3. **AlphaZero Training Pipeline**: A closed-loop system that generates training data through **Self-Play** without any human knowledge. It utilizes data augmentation (rotation/flipping) and iteratively updates the network parameters to minimize the loss between the neural network's predictions and the MCTS search results.

### Results

The implementation successfully trains an agent that masters Gomoku strategies on an $8\times8$ board. The results include:

* A **comprehensive report** (over 3 pages) detailing the AlphaZero framework, network architecture, and a deep analysis of the learning process.
* **Performance metrics** showing the rapid convergence of the training loss and the evolution of the agent's capability, evidenced by its win rate against a strong Pure MCTS baseline (achieving nearly 100% win rate).
* **Visualizations** including loss curves, win-rate evaluation charts, and **snapshots of evaluation games**, demonstrating the agent's mastery of advanced tactics such as "double-three" attacks and solid defensive formations.
