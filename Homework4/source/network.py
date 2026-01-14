"""
Copyright (C) 2026 Fu Tszkok

:module: network
:function: Defines the neural network architecture for the PPO agent, including the Transformer-based encoder, actor, and critic heads. Also includes the RolloutBuffer for experience storage.
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
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli


class ResBlock(nn.Module):
    """A standard residual block with two fully connected layers and layer normalization."""

    def __init__(self, d_model, dropout=0.1):
        """Initializes the residual block.
        :param d_model: The feature dimension of the input and output.
        :param dropout: The dropout probability.
        """
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Performs the forward pass of the residual block.
        :param x: The input tensor of shape (batch_size, d_model).
        :return: The output tensor after the residual connection.
        """
        identity = x

        out = self.fc1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.ln2(out)

        # Add the residual connection and apply final activation
        out += identity
        return F.relu(out)


class FeatureFusion(nn.Module):
    """A small MLP to fuse concatenated feature vectors into a single representation."""

    def __init__(self, input_dim, output_dim):
        """Initializes the fusion network.
        :param input_dim: The dimension of the concatenated input features.
        :param output_dim: The dimension of the fused output features.
        """
        super(FeatureFusion, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """Performs the forward pass.
        :param x: The concatenated input tensor.
        :return: The fused output tensor.
        """
        return self.net(x)


class GameEncoder(nn.Module):
    """Encodes the game state and history into a unified feature vector using a Transformer."""

    def __init__(self, state_dim, history_feature_dim, d_model=64, n_head=4, n_layers=2):
        """Initializes the game encoder components.
        :param state_dim: The dimensionality of the flat state vector.
        :param history_feature_dim: The feature dimensionality of a single history step.
        :param d_model: The main embedding dimension for the model.
        :param n_head: The number of attention heads in the Transformer.
        :param n_layers: The number of layers in the Transformer encoder.
        """
        super(GameEncoder, self).__init__()
        self.d_model = d_model
        # --- Component Networks ---
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(), ResBlock(d_model))
        self.history_fc = nn.Linear(history_feature_dim, d_model)
        # Using norm_first=True is a common practice for better stability in Transformers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, 30, d_model))  # Learnable positional embeddings
        self.fusion = FeatureFusion(d_model * 2, d_model)

    def forward(self, state, history, history_mask=None):
        """Performs the forward pass to get the combined feature representation.
        :param state: The current game state tensor.
        :param history: The game history tensor (sequence of past actions).
        :param history_mask: A boolean mask to ignore padding in the history sequence.
        :return: A fused feature tensor of shape (batch_size, d_model).
        """
        # 1. Encode the flat state vector
        s_emb = self.state_encoder(state)
        seq_len = history.size(1)

        # 2. Embed the history sequence and add positional information
        h_emb = self.history_fc(history) + self.pos_emb[:, :seq_len, :]
        h_out = self.transformer(h_emb, src_key_padding_mask=history_mask)

        # 3. Create a single context vector from the history sequence
        if history_mask is not None:
            # Use masking to compute a masked average pool, ignoring padded elements
            mask_expanded = (~history_mask).unsqueeze(-1).float()
            h_context = (h_out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            # If no mask, perform a simple mean pooling
            h_context = h_out.mean(dim=1)

        # 4. Concatenate state and history context and fuse them
        combined = torch.cat([s_emb, h_context], dim=1)
        features = self.fusion(combined)

        return features


class DoubtActor(nn.Module):
    """The actor head that decides whether to doubt the previous player."""

    def __init__(self, d_model):
        """Initializes the doubt actor.
        :param d_model: The input feature dimension from the encoder.
        """
        super(DoubtActor, self).__init__()
        self.trunk = nn.Sequential(ResBlock(d_model), ResBlock(d_model), ResBlock(d_model))
        self.head = nn.Linear(d_model, 2)  # Output logits for two actions: [Don't Doubt, Doubt]

    def forward(self, features):
        """Predicts the logits for the doubt action.
        :param features: The encoded feature vector from GameEncoder.
        :return: Logits for the doubt action.
        """
        x = self.trunk(features)
        logits = self.head(x)
        return logits


class PlayActor(nn.Module):
    """The actor head that decides which cards to play from the hand."""

    def __init__(self, d_model, hand_size=5):
        """Initializes the play actor.
        :param d_model: The input feature dimension from the encoder.
        :param hand_size: The maximum number of cards in a player's hand.
        """
        super(PlayActor, self).__init__()
        self.trunk = nn.Sequential(ResBlock(d_model), ResBlock(d_model), ResBlock(d_model))
        self.head = nn.Linear(d_model, hand_size)  # Output logits for each card slot

    def forward(self, features, hand_mask=None):
        """Predicts the logits for playing each card.
        :param features: The encoded feature vector from GameEncoder.
        :param hand_mask: A binary mask to prevent selecting empty card slots.
        :return: Logits for playing each card in hand.
        """
        x = self.trunk(features)
        logits = self.head(x)
        if hand_mask is not None:
            # Apply mask to make choosing non-existent cards extremely unlikely
            logits = logits.masked_fill(hand_mask == 0, -1e9)
        return logits


class Critic(nn.Module):
    """The critic head that estimates the state value and auxiliary tasks."""

    def __init__(self, d_model):
        """Initializes the critic.
        :param d_model: The input feature dimension from the encoder.
        """
        super(Critic, self).__init__()
        self.trunk = nn.Sequential(ResBlock(d_model), ResBlock(d_model), ResBlock(d_model), ResBlock(d_model))
        self.value_head = nn.Linear(d_model, 1)  # Main value prediction
        # Auxiliary heads can help with representation learning
        self.aux_next_player = nn.Linear(d_model, 4)  # Predicts who plays next
        self.aux_survival = nn.Linear(d_model, 4)  # Predicts survival probabilities

    def forward(self, features):
        """Predicts the state value and auxiliary targets.
        :param features: The encoded feature vector from GameEncoder.
        :return: A tuple `(value, pred_next_player, pred_survival)`.
        """
        x = self.trunk(features)
        value = self.value_head(x)
        pred_next_player = self.aux_next_player(x)
        pred_survival = self.aux_survival(x)
        return value, pred_next_player, pred_survival


class LiarBarAgentNetwork(nn.Module):
    """The complete PPO agent network, combining the encoder, actors, and critic."""

    def __init__(self, state_dim, history_dim, d_model=64):
        """Initializes the full agent network.
        :param state_dim: The dimensionality of the flat state vector.
        :param history_dim: The feature dimensionality of a single history step.
        :param d_model: The main embedding dimension for the model.
        """
        super(LiarBarAgentNetwork, self).__init__()
        self.encoder = GameEncoder(state_dim, history_dim, d_model=d_model)
        self.doubt_actor = DoubtActor(d_model)
        self.play_actor = PlayActor(d_model)
        self.critic = Critic(d_model)

    def forward(self):
        """This method is not intended for direct use.
        Use the `act` or `evaluate` methods instead.
        """
        raise NotImplementedError("Use specific methods (act/evaluate) instead.")

    def act(self, state, history, hand_mask, history_mask=None):
        """Selects an action for the agent to take (for inference).
        :param state: The current game state tensor.
        :param history: The game history tensor.
        :param hand_mask: A mask for the agent's hand.
        :param history_mask: A mask for the history sequence.
        :return: A tuple `(action_doubt, action_play, doubt_dist, play_dist)`.
        """
        features = self.encoder(state, history, history_mask)

        doubt_logits = self.doubt_actor(features)
        doubt_dist = Categorical(logits=doubt_logits)
        action_doubt = doubt_dist.sample()

        play_logits = self.play_actor(features, hand_mask)
        play_dist = Bernoulli(logits=play_logits)  # Bernoulli for multi-label (play multiple cards)
        action_play = play_dist.sample()

        return action_doubt, action_play, doubt_dist, play_dist

    def evaluate(self, state, history, action_doubt, action_play, hand_mask, history_mask=None):
        """Evaluates given states and actions (for PPO training).
        :param state: A batch of game state tensors.
        :param history: A batch of game history tensors.
        :param action_doubt: A batch of doubt actions taken.
        :param action_play: A batch of play actions taken.
        :param hand_mask: A batch of hand masks.
        :param history_mask: A batch of history masks.
        :return: A tuple `(doubt_log_prob, play_log_prob, value, dist_entropy, aux_np, aux_surv)`.
        """
        features = self.encoder(state, history, history_mask)

        # Get action distributions
        doubt_logits = self.doubt_actor(features)
        play_logits = self.play_actor(features, hand_mask)
        doubt_dist = Categorical(logits=doubt_logits)
        play_dist = Bernoulli(logits=play_logits)

        # Calculate log probabilities of the taken actions
        doubt_log_prob = doubt_dist.log_prob(action_doubt)
        play_log_prob = play_dist.log_prob(action_play).sum(dim=-1)

        # Calculate entropy for the exploration bonus
        dist_entropy = doubt_dist.entropy() + play_dist.entropy().sum(dim=-1)

        # Get state value and auxiliary predictions from the critic
        value, aux_np, aux_surv = self.critic(features)

        return doubt_log_prob, play_log_prob, value, dist_entropy, aux_np, aux_surv


class RolloutBuffer:
    """Stores trajectories of experience (state, action, reward, etc.) for PPO updates."""

    def __init__(self):
        """Initializes empty lists to store trajectory data."""
        self.actions_doubt = []
        self.actions_play = []
        self.states = []
        self.histories = []
        self.hand_masks = []
        self.history_masks = []
        self.logprobs_doubt = []
        self.logprobs_play = []
        self.rewards = []
        self.dones = []
        self.aux_next_player = []
        self.aux_survival = []

    def add(self, state, history, hand_mask, history_mask, action_doubt, action_play, logprob_doubt, logprob_play, reward, done, aux_np=None, aux_surv=None):
        """Adds a single step of experience to the buffer."""
        self.states.append(state)
        self.histories.append(history)
        self.hand_masks.append(hand_mask)
        self.history_masks.append(history_mask if history_mask is not None else None)
        self.actions_doubt.append(action_doubt)
        self.actions_play.append(action_play)
        self.logprobs_doubt.append(logprob_doubt)
        self.logprobs_play.append(logprob_play)
        self.rewards.append(reward)
        self.dones.append(done)

        if aux_np is not None: self.aux_next_player.append(aux_np)
        if aux_surv is not None: self.aux_survival.append(aux_surv)

    def clear(self):
        """Clears all stored data from the buffer, typically after a policy update."""
        del self.actions_doubt[:]
        del self.actions_play[:]
        del self.states[:]
        del self.histories[:]
        del self.hand_masks[:]
        del self.history_masks[:]
        del self.logprobs_doubt[:]
        del self.logprobs_play[:]
        del self.rewards[:]
        del self.dones[:]
        del self.aux_next_player[:]
        del self.aux_survival[:]

    def get_trajectory(self):
        """Retrieves and formats the stored data for the PPO update.
        :return: A dictionary containing trajectory data as NumPy arrays and PyTorch tensors.
        """
        # Note: Tensors are stacked to create a batch dimension.
        return {
            "states": np.array(self.states),
            "histories": np.array(self.histories),
            "hand_masks": torch.stack(self.hand_masks, dim=0),
            "history_masks": self.history_masks,
            "actions_doubt": np.array(self.actions_doubt),
            "actions_play": np.array(self.actions_play),
            "logprobs_doubt": torch.stack(self.logprobs_doubt),
            "logprobs_play": torch.stack(self.logprobs_play),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32)
        }
