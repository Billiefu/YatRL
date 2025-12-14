"""
Copyright (C) 2025 Fu Tszkok

:module: train
:function: Manages the AlphaZero training pipeline, including self-play with Dirichlet noise, Pitting (Best vs Current), and model optimization.
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
        """Initializes the training pipeline parameters, models, and components.
        :param init_model: Path to a pretrained model file (optional).
        """
        # --- 1. Game & Board Configuration ---
        # Increased board size to 10x10 for higher complexity.
        self.board_width = 10
        self.board_height = 10
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = self.board

        # --- 2. Training Hyperparameters ---
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0        # Adaptive multiplier for learning rate decay
        self.temp = 1.0                 # Temperature parameter for MCTS exploration (tau)
        self.n_playout = 400            # Number of MCTS simulations per move during training
        self.c_puct = 5                 # Exploration constant for PUCT formula
        self.buffer_size = 10000        # Experience replay buffer size
        self.batch_size = 512           # Mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.play_batch_size = 1        # Number of self-play games per iteration
        self.epochs = 5                 # Number of training epochs per batch of new data
        self.check_freq = 50            # Frequency of evaluation/pitting (in iterations)
        self.game_batch_num = 3000      # Total number of training iterations (increased for larger board)

        # --- 3. Model & Device Setup ---
        self.save_dir = './result/'
        self.best_model_path = os.path.join(self.save_dir, 'best_policy.pth')

        # Auto-detect computation device (CUDA/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training pipeline initialized on device: {self.device}")

        # Initialize the 'Current' Policy-Value Network (The challenger)
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, num_res_blocks=4)

        # Initialize the 'Best' Policy-Value Network (The champion/opponent)
        self.best_policy_net = PolicyValueNet(self.board_width, self.board_height, num_res_blocks=4)

        if init_model:
            print(f"Loading pretrained model from {init_model}...")
            state_dict = torch.load(init_model, map_location=self.device)
            self.policy_value_net.load_state_dict(state_dict)
            self.best_policy_net.load_state_dict(state_dict)
        else:
            print("Initializing new models with deeper ResNet structure...")
            # If starting from scratch, save the initial random model as the 'best' one
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            torch.save(self.policy_value_net.state_dict(), self.best_model_path)
            self.best_policy_net.load_state_dict(self.policy_value_net.state_dict())

        # Wrap networks for easy MCTS inference
        self.policy_value_net_wrapper = NetWrapper(self.policy_value_net)
        self.best_model_wrapper = NetWrapper(self.best_policy_net)

        # Initialize MCTS player for self-play using the CURRENT model
        self.mcts_player = MCTS(self.policy_value_net_wrapper.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        # Tracking metrics
        self.loss_records = []
        self.win_rate_records = []

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
        """Runs self-play games using the Current Model to collect training data.
        :param n_games: The number of games to play.
        :return: A list of (state, probabilities, winner) tuples representing the game history.
        """
        for i in range(n_games):
            self.board.init_board()
            self.mcts_player.update_with_move(-1) # Reset search tree

            states, mcts_probs, current_players = [], [], []

            while True:
                # Get action probabilities from MCTS
                acts, probs = self.mcts_player.get_move_probs(self.board, temp=self.temp)

                # This is crucial for AlphaZero to discover new strategies.
                # 0.3 is the noise weight (epsilon), 0.75 is the prior weight.
                p_dirichlet = np.random.dirichlet(0.3 * np.ones(len(probs)))
                probs = 0.75 * probs + 0.25 * p_dirichlet

                # Normalize probabilities again to ensure sum is 1.0
                probs /= np.sum(probs)

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

    def evaluate_against_best_model(self, n_games=10):
        """Runs the 'Pitting' process: Current Model vs. Best Model.
        :param n_games: Number of games to play for the duel.
        :return: The win rate of the Current Model (0.0 to 1.0).
        """
        # 1. Setup Players
        # Current Model Player (Challenger)
        current_mcts_player = MCTS(self.policy_value_net_wrapper.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        # Best Model Player (Champion) - Set to evaluation mode
        self.best_policy_net.eval()
        best_mcts_player = MCTS(self.best_model_wrapper.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        win_cnt = {'Current': 0, 'Best': 0, 'Tie': 0}

        # 2. Play Games
        for i in range(n_games):
            # Alternate starting player to ensure fairness
            # start_player=0 -> Player 1 starts.
            # We assign Current Model to Player 1, Best Model to Player 2
            winner, _ = self.start_play(current_mcts_player, best_mcts_player, start_player=i % 2)

            if winner == 1:
                win_cnt['Current'] += 1
            elif winner == 2:
                win_cnt['Best'] += 1
            else:
                win_cnt['Tie'] += 1

        # Calculate Win Rate (Draws count as 0.5 win)
        win_rate = (win_cnt['Current'] + 0.5 * win_cnt['Tie']) / n_games
        print(f"Pitting Result: Current {win_cnt['Current']} - {win_cnt['Best']} Best (Ties: {win_cnt['Tie']})")
        return win_rate

    def evaluate_against_pure_mcts(self, n_games=6):
        """Evaluates the current policy against a pure MCTS baseline (for benchmarking).
        :param n_games: The number of games to play for evaluation.
        :return: A tuple (win_rate, played_boards_history).
        """
        current_mcts_player = MCTS(self.policy_value_net_wrapper.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        # Pure MCTS uses a uniform random policy rollout
        # Increased playouts for Pure MCTS to provide a stronger benchmark
        pure_mcts_player = MCTS(
            lambda board: (zip(board.available, [1 / len(board.available)] * len(board.available)), 0) if len(board.available) > 0 else ([], 0),
            c_puct=5, n_playout=2000)

        win_cnt = 0
        played_boards = []

        for i in range(n_games):
            winner, final_states = self.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2)
            played_boards.append(final_states)
            if winner == 1:
                win_cnt += 1
            elif winner == -1:
                win_cnt += 0.5 # Tie counts as 0.5

        return win_cnt / n_games, played_boards

    def start_play(self, player1, player2, start_player=0):
        """Plays a single game between two MCTS players (helper function).
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
            # Use deterministic policy (temp close to 0) for evaluation
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
                # 1. Self-Play: Collect data using the Current Model (with noise)
                play_data = self.collect_selfplay_data(self.play_batch_size)
                self.data_buffer.extend(play_data)

                # 2. Policy Update: Train the network if enough data is collected
                if len(self.data_buffer) > self.batch_size:
                    loss = self.policy_update()
                    self.loss_records.append((i + 1, loss))
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'buffer': len(self.data_buffer)})
                else:
                    pbar.set_postfix({'buffer': len(self.data_buffer)})

                # 3. Evaluation & Pitting: Check performance periodically
                if (i + 1) % self.check_freq == 0:
                    pbar.write(f"\n--- Batch {i+1} Evaluation ---")

                    # A. Pitting: Challenge the Best Model
                    win_rate_vs_best = self.evaluate_against_best_model(n_games=10)

                    # If Current Model wins >= 55% of games, it becomes the new Best Model
                    if win_rate_vs_best >= 0.55:
                        pbar.write(f"New Best Model! Win rate: {win_rate_vs_best:.2f}")
                        # Save current model as the best model
                        torch.save(self.policy_value_net.state_dict(), self.best_model_path)
                        # Update the Best Model in memory
                        self.best_policy_net.load_state_dict(self.policy_value_net.state_dict())
                        self.best_model_wrapper = NetWrapper(self.best_policy_net)
                    else:
                        pbar.write(f"Challenge Failed. Win rate: {win_rate_vs_best:.2f}")

                    # B. Benchmarking: Check against Pure MCTS (for visualization)
                    win_rate_pure, played_boards = self.evaluate_against_pure_mcts(n_games=10)
                    self.win_rate_records.append((i + 1, win_rate_pure))
                    pbar.write(f"Win rate against Pure MCTS: {win_rate_pure:.2f}")

                    # Save plots and checkpoint
                    plot_loss(self.save_dir, self.loss_records)
                    plot_win_rate(self.save_dir, self.win_rate_records)
                    plot_games(self.save_dir, played_boards, i + 1, self.board_width, self.board_height)

                    # Also save current policy as a checkpoint
                    torch.save(self.policy_value_net.state_dict(), os.path.join(self.save_dir, 'current_policy.pth'))

                # 4. Learning Rate Decay: Reduce LR periodically
                if (i + 1) % 1000 == 0:
                    self.lr_multiplier *= 0.5
                    pbar.write(f"Learning Rate Multiplier decayed to {self.lr_multiplier}")


if __name__ == '__main__':
    pipeline = TrainPipeline()
    pipeline.run()
