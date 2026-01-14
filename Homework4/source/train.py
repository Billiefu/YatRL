"""
Copyright (C) 2026 Fu Tszkok

:module: train
:function: Orchestrates the two-phase training pipeline for the LiarBar agents. Phase 1 involves self-play to bootstrap each personality, and Phase 2 involves mixed-play where all personalities train against each other.
:author: Fu Tszkok
:date: 2026-01-14
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from agent import AggressiveBlufferAgent, AggressiveChallengerAgent, ConservativeAgent, RationalAgent
from bar import LiarBarEnv
from utils import CSVLogger, save_loss_curve, MODEL_DIR

# --- Training Hyperparameters ---
SELF_PLAY_EPISODES = 10000  # Episodes for each agent's self-play pre-training
MIXED_PLAY_EPISODES = 20000 # Episodes for the main mixed-personality training
UPDATE_TIMESTEP = 2048      # Number of steps to collect before a PPO update
LR = 0.0005                 # Learning rate for the Adam optimizer
GAMMA = 0.99                # Discount factor for future rewards

# --- Model & Environment Configuration ---
STATE_DIM = 33              # Dimensionality of the environment's state vector
HISTORY_DIM = 8             # Dimensionality of a single step in the history vector
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Agent Setup ---
AGENT_CLASSES = [AggressiveBlufferAgent, AggressiveChallengerAgent, ConservativeAgent, RationalAgent]
AGENT_NAMES = [cls.__name__.replace("Agent", "") for cls in AGENT_CLASSES]


def run_training_phase(phase_name, env, agents, total_episodes):
    """Executes a training phase for a given environment and set of agents.
    :param phase_name: A string name for the training phase (e.g., "Self-Play-Aggressive").
    :param env: The LiarBarEnv instance to train in.
    :param agents: A list of agent objects participating in the training.
    :param total_episodes: The total number of episodes to run for this phase.
    :return: A list of average loss values recorded during the phase.
    """
    print(f"\n>>> Starting Phase: {phase_name} ({total_episodes} episodes)")

    # --- Initialization ---
    fields = ['Step'] + [f"Agent{i}_Loss" for i in range(len(agents))] + ['Avg_Loss']
    csv_logger = CSVLogger(f"training_log_{phase_name}.csv", fields)
    env.log_callback = lambda x: None  # Disable verbose game logging for speed
    iterator = tqdm(range(1, total_episodes + 1), desc=f"Training {phase_name}", unit="ep")

    loss_history = []
    total_steps = 0

    # --- Main Training Loop ---
    for _ in iterator:
        obs = env.reset()
        state, history, hand_mask = obs

        # A single game episode loop
        while True:
            total_steps += 1
            curr_idx = env.current_player_idx
            curr_agent = agents[curr_idx]

            # 1. Action Selection
            action_doubt, action_play, log_d, log_p, _ = curr_agent.select_action(state, history)

            # 2. Check if the selected action constitutes a bluff (for reward shaping)
            is_bluff = False
            if action_doubt == 0:
                played_indices = [i for i, x in enumerate(action_play) if x == 1.0]
                real_cards = [curr_agent.hand[i] for i in played_indices if i < len(curr_agent.hand)]
                # A move is a bluff if any card played is not the table card or a Joker
                if any(card != env.table_card and card != 4 for card in real_cards):
                    is_bluff = True

            # 3. Environment Step
            next_obs, step_rewards, done = env.step(action_doubt, action_play)

            # 4. Store experience and apply reward shaping
            # First, add the transition to the current agent's buffer with a placeholder reward
            curr_agent.buffer.add(state, history, hand_mask, None, action_doubt, action_play, log_d, log_p, 0.0, done)
            # Then, distribute the rewards from the environment to all agents
            for agent_idx, raw_reward in step_rewards.items():
                agent = agents[agent_idx]
                shaped_reward = raw_reward
                # The agent who just acted receives its personality-driven shaped reward
                if agent_idx == curr_idx:
                    shaped_reward += agent.get_personality_reward(action_doubt, is_bluff, raw_reward)

                # Add the final reward to the last experience tuple in the buffer
                if len(agent.buffer.rewards) > 0:
                    agent.buffer.rewards[-1] += shaped_reward
                    if done: agent.buffer.dones[-1] = True # Mark the final step as done

            if done: break
            state, history, hand_mask = next_obs

            # 5. Periodic PPO Update
            # Perform a policy update for all agents after collecting a set number of steps
            if total_steps % UPDATE_TIMESTEP == 0:
                step_losses = {}
                loss_list = []
                for i, agent in enumerate(agents):
                    if len(agent.buffer.rewards) > 0:
                        loss = agent.update()
                        step_losses[f"Agent{i}_Loss"] = loss
                        loss_list.append(loss)
                    else:
                        step_losses[f"Agent{i}_Loss"] = 0.0

                if loss_list:
                    avg_loss = np.mean(loss_list)
                    loss_history.append(avg_loss)
                    # Log the loss data
                    log_row = {'Step': total_steps, 'Avg_Loss': avg_loss}
                    log_row.update(step_losses)
                    csv_logger.log(log_row)
                    iterator.set_postfix(loss=f"{avg_loss:.4f}")

    return loss_history


def main():
    """Orchestrates the entire two-phase training process."""
    print(f"Device: {DEVICE}")

    # --- PHASE 1: Self-Play Training ---
    # Goal: Train each agent personality against clones of itself to develop a baseline strategy.
    print("\n[PHASE 1] Self-Play Training")
    for cls, name in zip(AGENT_CLASSES, AGENT_NAMES):
        if name == "Rational":
            # Rational agent is a baseline and doesn't have a unique personality to pre-train.
            # It will inherit its initial weights from the Conservative agent.
            print(f"\nSkipping Phase 1 for {name} (will inherit from Conservative)...")
            continue

        print(f"\nTraining {name} vs {name}...")
        # Create an environment with 4 clones of the same agent class
        sp_agents = [cls(i, STATE_DIM, HISTORY_DIM, lr=LR, gamma=GAMMA, device=DEVICE) for i in range(4)]
        sp_env = LiarBarEnv(sp_agents)
        losses = run_training_phase(f"Self-Play-{name}", sp_env, sp_agents, SELF_PLAY_EPISODES)

        # Save the loss curve and the pre-trained model weights
        save_loss_curve(losses, name, f"loss_selfplay_{name}.png")
        torch.save(sp_agents[0].policy.state_dict(), os.path.join(MODEL_DIR, f"pretrained_{name}.pth"))

    # --- PHASE 2: Mixed-Play Training ---
    # Goal: Fine-tune the pre-trained agents by having them compete against each other.
    print("\n[PHASE 2] Mixed Training")
    mixed_agents = [
        AggressiveBlufferAgent(0, STATE_DIM, HISTORY_DIM, device=DEVICE),
        AggressiveChallengerAgent(1, STATE_DIM, HISTORY_DIM, device=DEVICE),
        ConservativeAgent(2, STATE_DIM, HISTORY_DIM, device=DEVICE),
        RationalAgent(3, STATE_DIM, HISTORY_DIM, device=DEVICE)
    ]

    # Load the pre-trained weights from Phase 1
    for i, agent in enumerate(mixed_agents):
        name = AGENT_NAMES[i]
        if name == "Rational":
            # The Rational agent starts with the Conservative agent's weights as a stable, neutral base.
            conservative_path = os.path.join(MODEL_DIR, "pretrained_Conservative.pth")
            if os.path.exists(conservative_path):
                agent.policy.load_state_dict(torch.load(conservative_path))
                print(f"Loaded Rational from Conservative weights: {conservative_path}")
            else:
                print("Warning: Conservative weights not found for Rational init. Starting from scratch.")
        else:
            path = os.path.join(MODEL_DIR, f"pretrained_{name}.pth")
            if os.path.exists(path):
                agent.policy.load_state_dict(torch.load(path))
                print(f"Loaded {name} from {path}")

    # Run the mixed-training phase
    mixed_env = LiarBarEnv(mixed_agents)
    losses = run_training_phase("Mixed-Training", mixed_env, mixed_agents, MIXED_PLAY_EPISODES)

    # Save the final loss curve and the final model weights for each agent
    save_loss_curve(losses, "Mixed", "loss_mixed.png")
    for i, agent in enumerate(mixed_agents):
        torch.save(agent.policy.state_dict(), os.path.join(MODEL_DIR, f"final_{AGENT_NAMES[i]}.pth"))

    print("\nTraining Complete. Please run test.py for evaluation.")


if __name__ == "__main__":
    # Script entry point
    main()
