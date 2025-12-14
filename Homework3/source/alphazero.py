"""
Copyright (C) 2025 Fu Tszkok

:module: alphazero
:function: Implements the Policy-Value Network (ResNet-based) and training wrapper for AlphaZero.
:author: Fu Tszkok
:date: 2025-12-08
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Implements a standard residual block for the ResNet architecture."""

    def __init__(self, channels):
        """Initializes the residual block.
        :param channels: The number of input and output channels.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Performs the forward pass of the residual block.
        :param x: Input tensor.
        :return: Output tensor after residual connection and activation.
        """
        # Save the input for the residual connection
        residual = x

        # First convolution block: Conv -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))

        # Second convolution block: Conv -> BN
        out = self.bn2(self.conv2(out))

        # Add the residual connection: F(x) + x
        out += residual

        # Apply final ReLU activation
        out = F.relu(out)
        return out


class PolicyValueNet(nn.Module):
    """Implements the Policy-Value Network used in AlphaZero."""

    def __init__(self, board_width, board_height, num_res_blocks=4, num_channels=64):
        """Initializes the Policy-Value Network.
        :param board_width: The width of the game board.
        :param board_height: The height of the game board.
        :param num_res_blocks: The number of residual blocks in the backbone.
        :param num_channels: The number of channels in the convolutional layers.
        """
        super(PolicyValueNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # Common input layer
        self.conv_input = nn.Conv2d(4, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head: Output logic probabilities for moves
        self.act_conv1 = nn.Conv2d(num_channels, 4, kernel_size=1, bias=False)
        self.act_bn1 = nn.BatchNorm2d(4)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # Value head: Output a scalar value [-1, 1] evaluation of the position
        self.val_conv1 = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.val_bn1 = nn.BatchNorm2d(2)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        """Performs the forward pass to get action probabilities and state value.
        :param state_input: The input state tensor (N, 4, H, W).
        :return: A tuple (log_act_probs, value).
        """
        # Process input through the common convolutional layer
        x = F.relu(self.bn_input(self.conv_input(state_input)))

        # Pass through the residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head forward pass
        x_act = F.relu(self.act_bn1(self.act_conv1(x)))
        x_act = x_act.view(x_act.size(0), -1)
        # Use log_softmax for numerical stability in loss calculation
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # Value head forward pass
        x_val = F.relu(self.val_bn1(self.val_conv1(x)))
        x_val = x_val.view(x_val.size(0), -1)
        x_val = F.relu(self.val_fc1(x_val))
        # Use tanh to map value to [-1, 1]
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


class NetWrapper(object):
    """Wraps the PolicyValueNet to handle training steps and data conversion."""

    def __init__(self, policy_value_net):
        """Initializes the network wrapper.
        :param policy_value_net: An instance of PolicyValueNet.
        """
        self.policy_value_net = policy_value_net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_value_net.to(self.device)

    def policy_value_fn(self, board):
        """Evaluates the current board state to get action probabilities and value.
        :param board: The current game board object.
        :return: A tuple (legal_act_probs, value) where legal_act_probs is a list of (move, prob).
        """
        legal_positions = board.available

        # Prepare the input state: (1, 4, Width, Height)
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, board.width, board.height))

        # Convert numpy array to torch tensor and move to device
        input_tensor = torch.from_numpy(current_state).float().to(self.device)

        # Set network to evaluation mode (disable dropout/batchnorm updates)
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(input_tensor)

        # Convert log probabilities back to probabilities
        act_probs = np.exp(log_act_probs.cpu().numpy().flatten())

        # Filter out illegal moves and pair moves with their probabilities
        legal_act_probs = zip(legal_positions, act_probs[legal_positions])

        value = value.item()

        return legal_act_probs, value

    def train_step(self, state_batch, mcts_probs_batch, winner_batch, lr=0.002):
        """Performs a single training step.
        :param state_batch: A batch of board states.
        :param mcts_probs_batch: A batch of target action probabilities from MCTS.
        :param winner_batch: A batch of actual game outcomes (winners).
        :param lr: The learning rate.
        :return: A tuple (loss, policy_loss, value_loss).
        """
        self.policy_value_net.train()
        optimizer = torch.optim.Adam(self.policy_value_net.parameters(), lr=lr, weight_decay=1e-4)

        # Convert batches to tensors and move to device
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs_batch = torch.FloatTensor(np.array(mcts_probs_batch)).to(self.device)
        winner_batch = torch.FloatTensor(np.array(winner_batch)).to(self.device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        log_act_probs, value = self.policy_value_net(state_batch)

        # Value loss: Mean Squared Error between predicted value and actual winner
        value_loss = F.mse_loss(value.view(-1), winner_batch)

        # Policy loss: Cross-Entropy (Negative Log Likelihood)
        # Loss = - sum(target_p * log(predicted_p))
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_act_probs, 1))

        # Total loss combines value and policy loss
        loss = value_loss + policy_loss

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()
