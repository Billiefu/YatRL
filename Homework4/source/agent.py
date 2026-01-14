"""
Copyright (C) 2026 Fu Tszkok

:module: agent
:function: Implements the PPO-based BaseAgent and its personality-driven subclasses for the LiarBar game.
:author: Fu Tszkok
:date: 2026-01-14
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from entity import *
from network import LiarBarAgentNetwork, RolloutBuffer


class BaseAgent:
    """Implements the core logic for a PPO-based agent, including the policy network, rollout buffer, and update mechanism."""

    def __init__(self, agent_id, state_dim, history_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, device=None, d_model=64):
        """Initializes the agent's components.
        :param agent_id: The unique identifier for the agent.
        :param state_dim: The dimensionality of the environment's state space.
        :param history_dim: The dimensionality of the action history space.
        :param lr: The learning rate for the Adam optimizer.
        :param gamma: The discount factor for future rewards.
        :param K_epochs: The number of update epochs to run per policy update.
        :param eps_clip: The clipping parameter for the PPO surrogate objective.
        :param device: The PyTorch device to run the models on ('cpu' or 'cuda').
        :param d_model: The dimensionality of the model's internal embeddings.
        """
        # --- 1. Agent Identification & RL Hyperparameters ---
        self.id = agent_id
        self.gamma = gamma
        self.device = device
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

        # --- 2. Game-Specific State ---
        self.hand = []
        self.gun = Gun()
        self.is_alive = True

        # --- 3. Policy Network & Optimizer ---
        self.policy = LiarBarAgentNetwork(state_dim, history_dim, d_model=d_model)
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # --- 4. Experience Replay & Loss Functions ---
        self.buffer = RolloutBuffer()
        self.MseLoss = nn.MSELoss()
        self.CrossEntropy = nn.CrossEntropyLoss()

    def reset(self):
        """Resets the agent's state for a new episode."""
        self.hand = []
        self.gun.reset()
        self.is_alive = True

    def get_hand_mask(self):
        """Creates a binary mask indicating which hand slots are occupied.
        :return: A PyTorch tensor mask of shape (5,) with 1s for occupied slots.
        """
        mask = torch.zeros(5)
        for i in range(len(self.hand)):
            mask[i] = 1
        return mask

    def select_action(self, state, history, history_mask=None):
        """Selects an action based on the current state and history using the policy network.
        :param state: The current environment state.
        :param history: The history of past actions.
        :param history_mask: A mask to ignore padding in the history tensor.
        :return: A tuple containing (doubt_action, play_action, doubt_logprob, play_logprob, hand_mask).
        """
        # Set network to evaluation mode for inference
        with torch.no_grad():
            # Prepare inputs: convert to tensors, add batch dimension, move to device
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            history = torch.FloatTensor(history).unsqueeze(0).to(self.device)
            hand_mask = self.get_hand_mask().unsqueeze(0).to(self.device)

            if history_mask is not None:
                history_mask = torch.BoolTensor(history_mask).unsqueeze(0).to(self.device)

            # Get actions and distributions from the policy network
            a_doubt, a_play, doubt_dist, play_dist = self.policy.act(state, history, hand_mask, history_mask)

            a_play_list = a_play.squeeze(0).tolist()
            doubt_item = a_doubt.item()

            # Edge case: If the policy outputs no action (doesn't doubt and doesn't play),
            # force the most likely 'play' action to prevent illegal moves.
            if doubt_item == 0 and sum(a_play_list) == 0:
                logits = play_dist.logits.squeeze(0)
                max_idx = torch.argmax(logits).item()
                a_play[0, max_idx] = 1.0
                a_play_list[max_idx] = 1.0

            # Calculate log probabilities for the chosen actions
            doubt_logprob = doubt_dist.log_prob(a_doubt)
            play_logprob = play_dist.log_prob(a_play).sum(dim=-1)

        return a_doubt.item(), a_play_list, doubt_logprob, play_logprob, hand_mask

    def update(self):
        """Performs the PPO policy update using data from the rollout buffer.
        :return: The average total loss over all training epochs.
        """
        # 1. Fetch Trajectory Data
        traj = self.buffer.get_trajectory()

        # Skip update if trajectory is too short
        if len(traj["states"]) <= 1:
            self.buffer.clear()
            return 0.0

        # 2. Data Preparation: Convert lists to tensors and move to device
        old_states = torch.FloatTensor(traj["states"]).to(self.device)
        old_histories = torch.FloatTensor(traj["histories"]).to(self.device)
        old_hand_masks = traj["hand_masks"].to(self.device)

        if traj["history_masks"] is not None and len(traj["history_masks"]) > 0 and traj["history_masks"][0] is not None:
            old_history_masks = torch.BoolTensor(np.stack(traj["history_masks"])).to(self.device)
        else:
            old_history_masks = None

        old_actions_doubt = torch.LongTensor(traj["actions_doubt"]).to(self.device)
        old_actions_play = torch.FloatTensor(traj["actions_play"]).to(self.device)
        old_logprobs_doubt = traj["logprobs_doubt"].to(self.device)
        old_logprobs_play = traj["logprobs_play"].to(self.device)

        rewards = torch.FloatTensor(traj["rewards"]).to(self.device)
        dones = torch.FloatTensor(traj["dones"]).to(self.device)

        # 3. Calculate Monte Carlo Returns (Rewards-to-Go)
        returns = []
        discounted_reward = 0
        # Iterate backwards through rewards and dones to calculate discounted returns
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_reward = 0  # Reset at the end of an episode
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # 4. Normalize Returns for training stability
        if returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)  # Add epsilon for numerical stability
        else:
            returns = returns - returns.mean()

        # --- 5. PPO Update Loop ---
        dataset_size = old_states.size(0)
        batch_size = 256
        total_loss_sum = 0
        update_count = 0

        for _ in range(self.K_epochs):
            # Create random permutations for mini-batch sampling
            indices = torch.randperm(dataset_size).to(self.device)

            for start_ind in range(0, dataset_size, batch_size):
                end_ind = start_ind + batch_size
                mb_inds = indices[start_ind:end_ind]

                if len(mb_inds) < 2:
                    continue  # Skip if batch is too small

                mb_hist_mask = old_history_masks[mb_inds] if old_history_masks is not None else None

                # A. Evaluate old actions and states with the current policy
                logprobs_doubt, logprobs_play, state_values, dist_entropy, aux_np, aux_surv = self.policy.evaluate(
                    old_states[mb_inds], old_histories[mb_inds], old_actions_doubt[mb_inds],
                    old_actions_play[mb_inds], old_hand_masks[mb_inds], mb_hist_mask
                )

                state_values = state_values.view(-1)
                mb_returns = returns[mb_inds]

                # B. Calculate Advantages (A = R - V)
                advantages = mb_returns - state_values.detach()

                # C. Normalize Advantages
                if advantages.std() > 1e-5:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
                else:
                    advantages = advantages - advantages.mean()

                # D. Calculate Policy Loss for 'Doubt' action (Clipped Surrogate Objective)
                mb_old_log_doubt = old_logprobs_doubt[mb_inds]
                ratios_doubt = torch.exp(logprobs_doubt - mb_old_log_doubt)
                surr1_d = ratios_doubt * advantages
                surr2_d = torch.clamp(ratios_doubt, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss_doubt = -torch.min(surr1_d, surr2_d).mean()

                # E. Calculate Policy Loss for 'Play' action
                mb_old_log_play = old_logprobs_play[mb_inds]
                ratios_play = torch.exp(logprobs_play - mb_old_log_play)
                surr1_p = ratios_play * advantages
                surr2_p = torch.clamp(ratios_play, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # The 'play' loss is only valid if the agent did not 'doubt'
                play_valid_mask = (old_actions_doubt[mb_inds] == 0).float()
                loss_play = (-torch.min(surr1_p, surr2_p) * play_valid_mask).sum() / (play_valid_mask.sum() + 1e-5)

                # F. Calculate Value Loss and Entropy Bonus
                loss_value = self.MseLoss(state_values, mb_returns)
                loss_entropy = -dist_entropy.mean() # Encourages exploration

                # G. Combine losses and perform backpropagation
                loss_total = loss_doubt + loss_play + 0.5 * loss_value + 0.01 * loss_entropy

                self.optimizer.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss_sum += loss_total.item()
                update_count += 1

        # 6. Clean up buffer for the next iteration
        self.buffer.clear()
        return total_loss_sum / update_count if update_count > 0 else 0.0

    def get_personality_reward(self, action_type, is_bluff, env_reward):
        """Defines the personality-driven reward shaping mechanism. Must be implemented by subclasses.
        :param action_type: The type of action taken (0 for play, 1 for doubt).
        :param is_bluff: A boolean indicating if the 'play' action was a bluff.
        :param env_reward: The raw reward received from the environment.
        :return: A float representing the shaped reward bonus.
        """
        raise NotImplementedError("Subclasses must implement personality reward.")


class AggressiveBlufferAgent(BaseAgent):
    """An agent that is rewarded for bluffing and playing aggressively."""
    def get_personality_reward(self, action_type, is_bluff, env_reward):
        bonus = 0
        if action_type == 0:  # If the agent plays a card
            bonus += 0.2  # Small reward for being active
            if is_bluff:
                bonus += 0.2  # Extra bonus for bluffing
        elif action_type == 1:  # If the agent challenges
            bonus -= 0.5  # Penalty for being passive/challenging
        return bonus


class AggressiveChallengerAgent(BaseAgent):
    """An agent that is rewarded for challenging opponents."""
    def get_personality_reward(self, action_type, is_bluff, env_reward):
        bonus = 0
        if action_type == 1:  # If the agent challenges
            bonus += 0.5  # High reward for challenging
            if env_reward > 0:  # If the challenge was successful
                bonus += 2.0  # Massive bonus for a successful challenge
        else:
            pass  # No reward or penalty for playing a card
        return bonus


class ConservativeAgent(BaseAgent):
    """An agent that avoids risk and is rewarded for honest play."""
    def get_personality_reward(self, action_type, is_bluff, env_reward):
        bonus = 0
        death_prob = self.gun.get_survival_prob()
        # Scale penalty based on how risky the situation is
        risk_penalty = 2.0 if death_prob > 0.3 else 1.0

        if env_reward < 0:
            # Penalize any action that results in a negative outcome from the environment
            bonus -= 1.0 * risk_penalty

        if action_type == 0:  # If the agent plays a card
            if not is_bluff:
                bonus += 0.5  # Reward for playing honestly
            else:
                bonus -= 2.0  # Heavy penalty for bluffing
        return bonus


class RationalAgent(BaseAgent):
    """A baseline agent that uses no personality-based reward shaping."""
    def get_personality_reward(self, action_type, is_bluff, env_reward):
        # This agent relies purely on the environment's reward signal.
        return 0.0
