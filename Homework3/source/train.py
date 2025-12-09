"""
Copyright (C) 2025 Fu Tszkok

:module: train
:function: Manages the AlphaZero training pipeline, including self-play, data collection, and model optimization.
:author: Fu Tszkok
:date: 2025-12-08
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import os
import random
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from alphazero import PolicyValueNet, NetWrapper
from board import Board
from mcts import MCTS
from utils import plot_loss, plot_win_rate, plot_games


class TrainPipeline:
    """Controls the complete training loop for the AlphaZero algorithm."""

    def __init__(self, init_model=None):
        """Initializes the training pipeline parameters and components.
        :param init_model: Path to a pretrained model file (optional).
        """
        # Game parameters
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = self.board

        # Training hyperparameters
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0        # Adaptive multiplier for learning rate
        self.temp = 1.0                 # Temperature parameter for MCTS exploration
        self.n_playout = 400            # Number of MCTS simulations per move
        self.c_puct = 5                 # Exploration constant
        self.buffer_size = 10000        # Experience replay buffer size
        self.batch_size = 512           # Mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1        # Number of self-play games per iteration
        self.epochs = 5                 # Number of training epochs per batch of new data
        self.check_freq = 50            # Frequency of evaluation (in iterations)
        self.game_batch_num = 1500      # Total number of training iterations

        # Tracking and saving
        self.loss_records = []
        self.win_rate_records = []
        self.save_dir = './result/'

        # Initialize the Policy-Value Network
        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
            self.policy_value_net.load_state_dict(torch.load(init_model))
        else:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)

        self.policy_value_net_wrapper = NetWrapper(self.policy_value_net)
        # Initialize MCTS player for self-play
        self.mcts_player = MCTS(self.policy_value_net_wrapper.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

    def get_equi_data(self, play_data):
        """Augments the dataset by rotating and flipping the board.
        :param play_data: A list of tuples (state, mcts_prob, winner).
        :return: A list of augmented data tuples.
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            # Generate 4 rotations and their flipped counterparts
            for i in [1, 2, 3, 4]:
                # Rotate state and probabilities counter-clockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))

                # Flip horizontally and add to data
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """Runs self-play games to collect training data.
        :param n_games: The number of games to play.
        :return: A list of (state, probabilities, winner) tuples representing the game history.
        """
        for i in range(n_games):
            self.board.init_board()
            self.mcts_player.update_with_move(-1)

            states, mcts_probs, current_players = [], [], []

            while True:
                # Use high temperature for the first 30 moves to encourage exploration
                temp = self.temp if len(states) < 30 else 1e-3
                acts, probs = self.mcts_player.get_move_probs(self.board, temp=temp)

                # Store the move probabilities for the full board size
                move_probs = np.zeros(self.board_width * self.board_height)
                move_probs[list(acts)] = probs

                # Choose action based on the calculated probabilities
                move = np.random.choice(acts, p=probs)

                # Store current state data
                states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(self.board.current_player)

                # Perform the move
                self.board.do_move(move)
                self.mcts_player.update_with_move(move)

                # Check for game end
                end, winner = self.board.game_end()
                if end:
                    # Create the return value Z for each step in the game history
                    winners_z = np.zeros(len(current_players))
                    if winner != -1:
                        # +1 reward if the player at that step won, -1 otherwise
                        winners_z[np.array(current_players) == winner] = 1.0
                        winners_z[np.array(current_players) != winner] = -1.0

                    self.mcts_player.update_with_move(-1)
                    # Return augmented data (rotations/flips)
                    return self.get_equi_data(zip(states, mcts_probs, winners_z))

    def policy_update(self):
        """Updates the policy-value network using the collected data buffer.
        :return: The average loss over the training epochs.
        """
        loss_sum = 0
        for i in range(self.epochs):
            # Sample a mini-batch from the experience buffer
            mini_batch = random.sample(self.data_buffer, self.batch_size)
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]

            # Perform a training step
            loss, entropy, v_loss = self.policy_value_net_wrapper.train_step(state_batch, mcts_probs_batch, winner_batch, lr=self.learn_rate * self.lr_multiplier)
            loss_sum += loss
        return loss_sum / self.epochs

    def evaluate_policy(self, n_games=6):
        """Evaluates the current policy against a pure MCTS baseline.
        :param n_games: The number of games to play for evaluation.
        :return: A tuple (win_rate, played_boards_history).
        """
        current_mcts_player = MCTS(self.policy_value_net_wrapper.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        # Pure MCTS uses a uniform random policy rollout
        pure_mcts_player = MCTS(
            lambda board: (zip(board.available, [1 / len(board.available)] * len(board.available)), 0) if len(board.available) > 0 else ([], 0),
            c_puct=5, n_playout=1000)

        win_cnt = {'AlphaZero': 0, 'PureMCTS': 0, 'Tie': 0}
        played_boards = []

        for i in range(n_games):
            # Alternate starting player
            winner, final_states = self.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2)
            played_boards.append(final_states)
            if winner == 1:
                win_cnt['AlphaZero'] += 1
            elif winner == 2:
                win_cnt['PureMCTS'] += 1
            else:
                win_cnt['Tie'] += 1
        return win_cnt['AlphaZero'] / n_games, played_boards

    def start_play(self, player1, player2, start_player=0):
        """Plays a single game between two MCTS players.
        :param player1: The first player instance (assigned index 1).
        :param player2: The second player instance (assigned index 2).
        :param start_player: The index of the starting player (0 or 1).
        :return: A tuple (winner, board_states).
        """
        self.board.init_board(start_player)
        p1, p2 = self.board.players

        # Reset search trees
        player1.update_with_move(-1)
        player2.update_with_move(-1)

        players = {p1: player1, p2: player2}

        while True:
            player_in_turn = players[self.board.current_player]
            # Use low temperature for deterministic play during evaluation
            acts, probs = player_in_turn.get_move_probs(self.board, temp=1e-3)
            move = np.random.choice(acts, p=probs)

            self.board.do_move(move)

            # Sync both players' MCTS trees with the move made
            player1.update_with_move(move)
            player2.update_with_move(move)

            end, winner = self.board.game_end()
            if end:
                return winner, self.board.states.copy()

    def run(self):
        """Executes the main training loop."""
        print(f"Start training! Board size: {self.board_width}x{self.board_height}, Connect: {self.n_in_row}")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        with tqdm(range(self.game_batch_num), desc="Training") as pbar:
            for i in pbar:
                # 1. Self-Play: Collect data
                play_data = self.collect_selfplay_data(self.play_batch_size)
                self.data_buffer.extend(play_data)

                # 2. Policy Update: Train the network if enough data is collected
                if len(self.data_buffer) > self.batch_size:
                    loss = self.policy_update()
                    self.loss_records.append((i + 1, loss))
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'buffer': len(self.data_buffer)})
                else:
                    pbar.set_postfix({'buffer': len(self.data_buffer)})

                # 3. Evaluation: Check performance periodically
                if (i + 1) % self.check_freq == 0:
                    pbar.write("Evaluating policy...")

                    win_rate, played_boards = self.evaluate_policy(n_games=10)
                    self.win_rate_records.append((i + 1, win_rate))

                    pbar.write(f"Win rate against Pure MCTS: {win_rate:.2f}")

                    # Save plots and model
                    plot_loss(self.save_dir, self.loss_records)
                    plot_win_rate(self.save_dir, self.win_rate_records)
                    plot_games(self.save_dir, played_boards, i + 1, self.board_width, self.board_height)

                    pbar.write(f"Evaluation games saved to ./result/eval_batch_{i + 1}.png")
                    torch.save(self.policy_value_net.state_dict(), os.path.join(self.save_dir, 'current_policy.pth'))

                    if win_rate >= 0.95:
                        pbar.write("Win rate reached criteria, model performs excellently!")


if __name__ == '__main__':
    pipline = TrainPipeline()
    pipline.run()
