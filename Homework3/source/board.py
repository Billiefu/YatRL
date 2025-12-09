"""
Copyright (C) 2025 Fu Tszkok

:module: board
:function: Implements the game board logic for Gomoku (Five-in-a-Row) or similar grid games.
:author: Fu Tszkok
:date: 2025-12-08
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np


class Board(object):
    """Implements the board state and logic for an n-in-a-row game."""

    def __init__(self, width=8, height=8, n_in_row=5, start_player=0):
        """Initializes the board parameters.
        :param width: The width of the board.
        :param height: The height of the board.
        :param n_in_row: The number of stones in a row required to win.
        :param start_player: The index of the starting player (0 or 1).
        """
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.players = [1, 2]  # Player 1 and Player 2

        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('Board width and height can not be less than {}'.format(self.n_in_row))

        self.current_player = self.players[start_player]
        # Keep a list of available moves (linear indices)
        self.available = list(range(self.width * self.height))
        # Dictionary to store the state of the board {move_index: player_id}
        self.states = {}
        self.last_move = -1

    def init_board(self, start_player=0):
        """Resets the board state for a new game.
        :param start_player: The index of the starting player (0 or 1).
        """
        self.current_player = self.players[start_player]
        self.available = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """Converts a linear move index to a (row, col) location.
        :param move: The linear index of the move (0 to width*height - 1).
        :return: A list [row, col].
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """Converts a (row, col) location to a linear move index.
        :param location: A list or tuple [row, col].
        :return: The linear index of the move, or -1 if invalid.
        """
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """Returns the current board state as a 4-plane feature map for the neural network.
        :return: A NumPy array of shape (4, width, height).
                 - Plane 0: Current player's stones (1 for presence, 0 otherwise).
                 - Plane 1: Opponent's stones.
                 - Plane 2: The last move played.
                 - Plane 3: Color to play (all 1s if player 1's turn, else 0).
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(self.states.keys())), np.array(list(self.states.values()))
            # Separate moves by current player and opponent
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]

            # Populate Plane 0 (Current Player) and Plane 1 (Opponent)
            if len(move_curr) > 0:
                square_state[0][move_curr // self.width, move_curr % self.width] = 1.0
            if len(move_oppo) > 0:
                square_state[1][move_oppo // self.width, move_oppo % self.width] = 1.0

            # Populate Plane 2 with the last move position
            square_state[2][self.last_move // self.width, self.last_move % self.width] = 1.0

        # Populate Plane 3 based on whose turn it is (parity of total moves)
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0

        return square_state

    def do_move(self, move):
        """Executes a move on the board.
        :param move: The linear index of the move to play.
        """
        if move not in self.available:
            raise ValueError(f"Move {move} is not in available list: {self.available}")

        # Update board state
        self.states[move] = self.current_player
        self.available.remove(move)

        # Switch player
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.last_move = move

    def has_a_winner(self):
        """Checks if there is a winner on the board.
        :return: A tuple (has_winner, winner_id). Returns (False, -1) if no winner.
        """
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(self.states.keys())
        # Optimization: Impossible to win if total moves are fewer than 2*n - 1
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            # Check horizontal line: m, m+1, ..., m+n-1
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # Check vertical line: m, m+w, ..., m+(n-1)*w
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            # Check diagonal line (\): m, m+(w+1), ...
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            # Check anti-diagonal line (/): m, m+(w-1), ...
            if (w in range(width - n + 1) and h in range(n - 1, height) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Determines if the game has ended (win or draw).
        :return: A tuple (is_end, winner).
                 - winner is player_id if won, -1 if draw or not ended.
        """
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not self.available:
            # No moves left, it's a draw
            return True, -1
        return False, -1

    def get_current_player(self):
        """Returns the ID of the current player.
        :return: The player ID (1 or 2).
        """
        return self.current_player
