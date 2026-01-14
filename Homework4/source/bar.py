"""
Copyright (C) 2026 Fu Tszkok

:module: bar
:function: Implements the LiarBar game environment, managing game state, agent interactions, rules, and rewards.
:author: Fu Tszkok
:date: 2026-01-13
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import random
from collections import deque
import numpy as np
from entity import CARD_MAP, decode_cards


class LiarBarEnv:
    """Manages the complete game loop for LiarBar.
    This includes handling the deck, player turns, actions (playing cards and doubting),
    state transitions, and reward calculation.
    """

    def __init__(self, agents, history_len=20, log_callback=None):
        """Initializes the LiarBar environment.
        :param agents: A list of agent objects participating in the game.
        :param history_len: The maximum length of the action history to maintain.
        :param log_callback: An optional function for logging game events. Defaults to `print`.
        """
        # --- 1. Game Participants & Configuration ---
        self.agents = agents
        self.n_players = len(agents)
        self.history_len = history_len
        self.deck_config = [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6 + [4] * 2  # Card distribution

        # --- 2. Game State Variables ---
        self.table_card = None
        self.current_player_idx = 0
        self.round_history = deque(maxlen=history_len)
        self.last_move = None
        self.game_over = False
        self.winner = None

        # --- 3. Utilities ---
        self.log_callback = log_callback if log_callback else print

    def log(self, msg):
        """Logs a message using the provided callback function.
        :param msg: The message string to log.
        """
        if self.log_callback:
            self.log_callback(msg)

    def reset(self):
        """Resets the environment to a starting state for a new episode.
        :return: The initial observation tuple (state, history, hand_mask) for the first player.
        """
        # Reset and shuffle the deck
        deck = self.deck_config.copy()
        random.shuffle(deck)

        # Reset each agent's state, hand, gun, and buffer
        for agent in self.agents:
            agent.hand = [deck.pop() for _ in range(5)]
            agent.is_alive = True
            agent.gun.reset()
            agent.buffer.clear()

        # Set a new table card and randomly select a starting player
        self.table_card = random.randint(0, 3)
        self.current_player_idx = random.randint(0, self.n_players - 1)
        self.game_over = False
        self.winner = None
        self.last_move = None

        # Initialize the history buffer with empty records
        self.round_history.clear()
        for _ in range(self.history_len):
            self.round_history.append(np.zeros(8))

        return self._get_observation(self.current_player_idx)

    def step(self, action_doubt, action_play):
        """Executes one time step in the environment based on the current player's action.
        :param action_doubt: Integer (0 or 1) indicating if the agent doubts the last move.
        :param action_play: A list/tensor representing the cards played.
        :return: A tuple `(next_observation, rewards, done)`.
        """
        current_agent = self.agents[self.current_player_idx]

        # Initialize a small positive reward for all living agents to encourage survival
        step_rewards = {i: 0.01 for i in range(self.n_players) if self.agents[i].is_alive}

        if action_doubt == 1:
            # Handle the 'Doubt' action
            if self.last_move is None:
                # Penalize illegal doubt action (doubting when there's no previous move)
                step_rewards[self.current_player_idx] = -1.0
                self._record_history(self.current_player_idx, 1, 0, 0)
            else:
                self._resolve_doubt(step_rewards)
                self.last_move = None  # The doubt action clears the last move
        else:
            # Handle the 'Play Card(s)' action
            played_indices = [i for i, x in enumerate(action_play) if x == 1]
            played_cards = [current_agent.hand[i] for i in played_indices]
            count = len(played_cards)

            # Penalize and correct illegal play actions (e.g., playing 0 or >3 cards)
            if count < 1 or count > 3:
                step_rewards[self.current_player_idx] = -1.0
                if len(current_agent.hand) > 0:
                    idx = random.randint(0, len(current_agent.hand) - 1)
                    played_cards = [current_agent.hand[idx]]
                    played_indices = [idx]
                else:
                    played_cards = []  # No cards to play if hand is empty

            # Remove played cards from the agent's hand
            for i in sorted(played_indices, reverse=True):
                del current_agent.hand[i]

            self.last_move = {'player_idx': self.current_player_idx, 'cards': played_cards, 'count': len(played_cards)}
            self._record_history(self.current_player_idx, 0, len(played_cards), 0)

            # If player empties their hand, reward them and deal a new hand to continue play
            if len(current_agent.hand) == 0:
                step_rewards[self.current_player_idx] += 0.5
                current_agent.hand = [random.choice(self.deck_config) for _ in range(5)]

        # --- Check for Game End Condition ---
        alive_indices = [i for i, a in enumerate(self.agents) if a.is_alive]
        if len(alive_indices) <= 1:
            self.game_over = True
            if len(alive_indices) == 1:
                # A single survivor is the winner
                self.winner = alive_indices[0]
                step_rewards[self.winner] += 5.0  # Large reward for winning

            # Apply large penalty to all eliminated players
            for i in range(self.n_players):
                if not self.agents[i].is_alive:
                    current_val = step_rewards.get(i, 0.0)
                    step_rewards[i] = current_val - 5.0
            return None, step_rewards, True

        # --- Prepare for Next Turn ---
        self._next_turn()
        next_obs = self._get_observation(self.current_player_idx)
        return next_obs, step_rewards, False

    def _resolve_doubt(self, rewards):
        """Handles the logic when a player doubts the previous move.
        This includes determining if the previous player was lying and applying rewards/penalties.
        :param rewards: A dictionary of rewards to be modified in-place.
        """
        challenger_idx = self.current_player_idx
        prev_idx = self.last_move['player_idx']
        played_cards = self.last_move['cards']

        self.log(f"!!! P{challenger_idx} doubts P{prev_idx} !!!")

        # Determine if the previous player was lying (played a card that wasn't the table card or a wild card)
        is_liar = any(card != self.table_card and card != 4 for card in played_cards)

        if is_liar:
            # Case 1: The doubter was correct (Liar caught)
            self.log(f"  > LIAR CAUGHT! P{prev_idx} was lying (Played {decode_cards(played_cards)} on {CARD_MAP[self.table_card]}).")
            victim_idx = prev_idx
            rewards[challenger_idx] += 1.0
            rewards[prev_idx] -= 0.5
        else:
            # Case 2: The doubter was wrong (Previous player was honest)
            self.log(f"  > TRUTH! P{prev_idx} was honest. P{challenger_idx} plays Russian Roulette.")
            victim_idx = challenger_idx
            rewards[challenger_idx] -= 0.5
            rewards[prev_idx] += 1.0

        # The loser of the doubt must pull the trigger
        victim_agent = self.agents[victim_idx]
        is_dead = victim_agent.gun.pull_trigger()
        hist_result = -1 if is_dead else 1
        self._record_history(victim_idx, 1, 0, hist_result)

        if is_dead:
            self.log(f"  > BANG! P{victim_idx} is eliminated.")
            victim_agent.is_alive = False
            rewards[victim_idx] -= 5.0  # Heavy penalty for elimination
        else:
            self.log(f"  > CLICK. P{victim_idx} survives.")
            rewards[victim_idx] -= 0.5

    def _next_turn(self):
        """Advances the turn to the next living player."""
        steps = 0
        while steps < self.n_players:
            self.current_player_idx = (self.current_player_idx + 1) % self.n_players
            # Skips over any players that have been eliminated
            if self.agents[self.current_player_idx].is_alive:
                break
            steps += 1

    def _record_history(self, player_idx, action_type, count, result):
        """Encodes and records a game event into the round history buffer.
        :param player_idx: The index of the player who acted.
        :param action_type: The type of action (0 for play, 1 for doubt).
        :param count: The number of cards played (if action_type is 0).
        :param result: The outcome of a doubt (1 for success, -1 for failure).
        """
        p_vec = [0] * 4
        p_vec[player_idx] = 1
        info_vec = [action_type, count / 5.0, result, 0]  # Normalize count
        entry = np.array(p_vec + info_vec, dtype=np.float32)
        self.round_history.append(entry)

    def _get_observation(self, player_idx):
        """Constructs the observation (state and history) for a specific agent.
        :param player_idx: The index of the player for whom to generate the observation.
        :return: A tuple `(state_np, history_np, hand_mask)`.
        """
        agent = self.agents[player_idx]
        # --- State Vector Construction ---
        # The target card on the table
        target_vec = np.eye(5)[self.table_card]
        # Which player's turn it is
        active_vec = np.eye(4)[player_idx]
        # Survival probabilities for each player
        gun_vec = [a.gun.get_survival_prob() for a in self.agents]
        # Who is still alive
        alive_vec = [1 if a.is_alive else 0 for a in self.agents]
        # Normalized count of cards in each player's hand
        hand_counts = [len(a.hand) / 5.0 for a in self.agents]
        # The number of cards played in the last move
        last_move_vec = [0] * 3
        if self.last_move:
            count = self.last_move['count']
            if 1 <= count <= 3:
                last_move_vec[count - 1] = 1
        # The observing agent's own hand composition
        my_hand_vec = np.bincount(agent.hand, minlength=5)
        # The observing agent's own ID
        id_vec = np.eye(4)[player_idx]

        state_np = np.concatenate([
            target_vec, active_vec, gun_vec, alive_vec,
            hand_counts, last_move_vec, my_hand_vec.flatten(), id_vec.flatten()
        ]).astype(np.float32)

        history_np = np.array(self.round_history, dtype=np.float32)

        return state_np, history_np, agent.get_hand_mask()

    def print_state(self):
        """Prints a human-readable summary of the current game state."""
        self.log(f"\n>>> Table Target: [{CARD_MAP[self.table_card]}] | Turn: P{self.current_player_idx}")
        if self.last_move:
            prev = self.last_move['player_idx']
            cnt = self.last_move['count']
            self.log(f"    Last Play: P{prev} played {cnt} cards.")
        self.log("-" * 20)
