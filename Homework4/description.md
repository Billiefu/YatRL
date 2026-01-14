## HOMEWORK4: Liar's Bar: Reinforcement Through Deception

This project is an implementation for a reinforcement learning course assignment. The objective is to design intelligent agents to play the strategy game **Liar's Bar**, which uniquely integrates card-based bluffing mechanics with the high-stakes risk of **Russian Roulette**. This project explores how reinforcement learning agents balance the incentive for deception against the probability of survival in a non-stationary, imperfect-information environment.

### Problem Formulation (POSG)

The Liar's Bar environment is formally modeled as a multi-player, general-sum, **Partially Observable Stochastic Game (POSG)**. The key components are defined as follows:

*   **States**: The global state is hidden. Agents receive a **33-dimensional observation vector**, which includes one-hot encodings of the target card, the agent's private hand distribution, a summary of the previous player's action, and a crucial scalar representing the **Current Cylinder Risk** (probability of death).
*   **Actions**: The action space is **composite and discrete**, consisting of two decoupled branches: a binary decision to **Doubt** the previous player, and a multi-dimensional decision to **Play Cards** (selecting specific cards from the hand).
*   **Rewards**: A hybrid reward structure is used. It combines sparse environmental rewards (+5 for winning, -5 for elimination) with dense, **personality-based intrinsic rewards** (e.g., incentives for successful bluffing or aggressive doubting) to shape diverse agent behaviors.
*   **Transitions**: The model is highly stochastic. Beyond standard card dealing, the transition dynamics include an **irreversible elimination mechanism** triggered by the Russian Roulette, which monotonically decreases the number of agents and increases the lethality of the game over time.

### Implemented Algorithms

This repository provides the code to train a robust "Rational" agent using a curriculum learning approach, featuring the following core components:

1.  **Transformer-based Actor-Critic**: Unlike traditional MLPs, our network architecture incorporates a **Transformer Encoder** to process the sequence of historical actions. This allows the agent to use **Multi-Head Self-Attention** to identify opponent patterns and infer hidden information (e.g., predicting if an opponent is a bluffer based on past moves).
2.  **Proximal Policy Optimization (PPO)**: We employ PPO, a stable on-policy gradient method, to optimize the policy in this complex discrete action space, utilizing Generalized Advantage Estimation (GAE) to guide gradient updates.
3.  **Personality Shaping via Reward Engineering**: To prevent strategy homogenization in self-play, we pre-train baseline agents with distinct personas:
    *   **AggressiveBluffer**: Incentivized to lie frequently.
    *   **AggressiveChallenger**: Incentivized to doubt indiscriminately.
    *   **Conservative**: Incentivized to minimize risk and survive.
4.  **Curriculum Training Pipeline**: The training proceeds in two phases: **Homogeneous Self-Play** to solidify basic rules and personas, followed by **Heterogeneous Mixed-Play**, where a "Rational" agent (with no personality bias) learns to counter the pre-trained personalities and evolve a robust Nash Equilibrium strategy.

### Results

The implementation successfully trains a Rational agent that dominates in heterogeneous melees. The results include:

*   A **comprehensive report** (over 8 pages) detailing the POSG formulation, Transformer architecture, and the impact of personality shaping.
*   **Quantitative Metrics** illustrating the performance stratification, where the Rational agent achieves the highest **Score (1.78)** and **Strategy Success Rate (0.67)** compared to personality baselines.
*   **Training Dynamics Analysis** showing the stability of the mixed-training phase versus the high variance (loss spikes) observed in aggressive self-play.
*   **Qualitative Case Studies** analyzing game logs, demonstrating advanced tactical behaviors such as "Targeted Exploitation" (increasing doubt frequency against Bluffers) and "Risk Management" (avoiding action when cylinder lethality is critical).## HOMEWORK4: Liar's Bar
