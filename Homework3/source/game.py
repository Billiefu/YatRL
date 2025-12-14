"""
Copyright (C) 2025 Fu Tszkok

:module: game
:function: Provides a Pygame-based GUI for Human vs AlphaZero Gomoku. Handles game rendering, user input, and AI inference.
:author: Fu Tszkok
:date: 2025-12-08
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import sys
import pygame
import torch
import numpy as np
from board import Board
from alphazero import PolicyValueNet, NetWrapper
from mcts import MCTS


# --- Game Configuration ---
# IMPORTANT: These dimensions must match the trained model's configuration.
BOARD_WIDTH = 10
BOARD_HEIGHT = 10
N_IN_ROW = 5
MODEL_FILE = './result/best_policy.pth'  # Load the best model available

# AI Strength Configuration
# Higher playouts = Stronger AI but slower thinking time.
# 400 is fast, 1000 is strong, 2000+ is grandmaster level (on 10x10).
PLAYOUTS = 1000

# --- UI Configuration ---
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640
BG_COLOR = (219, 178, 126)              # Wood texture color
LINE_COLOR = (40, 40, 40)               # Grid line color
BLACK_STONE_COLOR = (10, 10, 10)
WHITE_STONE_COLOR = (240, 240, 240)
HIGHLIGHT_COLOR = (255, 50, 50)         # Red marker for the last move
HOVER_COLOR = (0, 0, 0, 100)            # Transparent shadow for mouse hover

# Layout calculations
MARGIN = 50
GRID_SIZE = (SCREEN_WIDTH - 2 * MARGIN) // (BOARD_WIDTH - 1)


class GomokuGUI:
    """Manages the Graphical User Interface for the game."""

    def __init__(self):
        """Initializes the Pygame window, game board, and AI player."""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AlphaZero Gomoku - [R] to Restart/Switch Side")

        # Initialize fonts
        self.font = pygame.font.SysFont('Arial', 20, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 32, bold=True)

        # Initialize Game Logic
        self.board = Board(width=BOARD_WIDTH, height=BOARD_HEIGHT, n_in_row=N_IN_ROW)
        self.mcts = self.load_ai()

        # Game State
        self.human_plays_black = True  # True: Human first (Black), False: AI first (Black)
        self.game_over = False
        self.winner = -1

        # If Human plays White (second), AI makes the first move immediately
        if not self.human_plays_black:
            self.ai_move()

    def load_ai(self):
        """Loads the pretrained PyTorch model and initializes the MCTS player.
        :return: An MCTS instance ready for inference.
        """
        print(f"Loading model from {MODEL_FILE}...")
        try:
            # Force CPU for inference to avoid CUDA overhead/compatibility issues in simple GUI
            device = torch.device('cpu')

            # Initialize network structure (Must match training: num_res_blocks=4)
            policy_value_net = PolicyValueNet(BOARD_WIDTH, BOARD_HEIGHT, num_res_blocks=4)

            # Load weights
            state_dict = torch.load(MODEL_FILE, map_location=device)
            policy_value_net.load_state_dict(state_dict)

            # Wrap network for MCTS
            net_wrapper = NetWrapper(policy_value_net, device='cpu')

            # Create MCTS player
            return MCTS(net_wrapper.policy_value_fn, c_puct=5, n_playout=PLAYOUTS)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Ensure train.py has run and saved the model.")
            sys.exit()

    def reset_game(self, switch_side=False):
        """Resets the game state and optionally switches the player's side.
        :param switch_side: If True, swaps Human/AI roles (Black/White).
        """
        if switch_side:
            self.human_plays_black = not self.human_plays_black

        # Reset board logic
        self.board.init_board()
        self.mcts.update_with_move(-1)
        self.game_over = False
        self.winner = -1

        print(f"Game Reset. Human plays {'Black' if self.human_plays_black else 'White'}.")

        # If AI is Black (First player), it moves immediately
        if not self.human_plays_black:
            self.draw_all()
            pygame.display.flip()
            self.ai_move()

    def get_coordinate(self, pos):
        """Converts screen pixel coordinates to board grid indices.
        :param pos: Tuple (x, y) from mouse event.
        :return: Tuple (row, col) or None if click is out of bounds.
        """
        x, y = pos
        # Use rounding logic to find the nearest grid intersection
        col = int((x - MARGIN + GRID_SIZE / 2) / GRID_SIZE)
        row = int((y - MARGIN + GRID_SIZE / 2) / GRID_SIZE)

        if 0 <= col < BOARD_WIDTH and 0 <= row < BOARD_HEIGHT:
            return row, col
        return None

    def draw_board(self):
        """Renders the wooden background and grid lines."""
        self.screen.fill(BG_COLOR)

        # Draw horizontal lines
        for i in range(BOARD_HEIGHT):
            start = (MARGIN, MARGIN + i * GRID_SIZE)
            end = (MARGIN + (BOARD_WIDTH - 1) * GRID_SIZE, MARGIN + i * GRID_SIZE)
            pygame.draw.line(self.screen, LINE_COLOR, start, end, 2)

        # Draw vertical lines
        for i in range(BOARD_WIDTH):
            start = (MARGIN + i * GRID_SIZE, MARGIN)
            end = (MARGIN + i * GRID_SIZE, MARGIN + (BOARD_HEIGHT - 1) * GRID_SIZE)
            pygame.draw.line(self.screen, LINE_COLOR, start, end, 2)

        # Draw the center star (Tian Yuan)
        center_x = MARGIN + (BOARD_WIDTH // 2) * GRID_SIZE
        if BOARD_WIDTH % 2 == 0: center_x -= GRID_SIZE // 2
        center_y = MARGIN + (BOARD_HEIGHT // 2) * GRID_SIZE
        if BOARD_HEIGHT % 2 == 0: center_y -= GRID_SIZE // 2
        pygame.draw.circle(self.screen, LINE_COLOR, (center_x, center_y), 5)

        # Draw coordinate labels (0, 1, 2...)
        for i in range(BOARD_WIDTH):
            label = self.font.render(str(i), True, LINE_COLOR)
            self.screen.blit(label, (MARGIN + i * GRID_SIZE - 5, MARGIN - 25))
            self.screen.blit(label, (MARGIN - 25, MARGIN + i * GRID_SIZE - 10))

    def draw_stones(self):
        """Renders all placed stones with a 3D-like lighting effect."""
        for move, player in self.board.states.items():
            h, w = self.board.move_to_location(move)
            x = MARGIN + w * GRID_SIZE
            y = MARGIN + h * GRID_SIZE

            color = BLACK_STONE_COLOR if player == 1 else WHITE_STONE_COLOR
            # Draw base stone
            pygame.draw.circle(self.screen, color, (x, y), GRID_SIZE // 2 - 2)

            # Draw highlight (simulate light source from top-left)
            # Ensure RGB values do not exceed 255
            bright_r = min(255, color[0] + 40)
            bright_g = min(255, color[1] + 40)
            bright_b = min(255, color[2] + 40)

            pygame.draw.circle(self.screen, (bright_r, bright_g, bright_b), (x - 5, y - 5), 5)

            # Mark the last move with a red indicator
            if move == self.board.last_move:
                pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, (x, y), 4)

    def draw_hover(self, pos):
        """Renders a transparent 'ghost' stone under the mouse cursor."""
        if self.game_over: return

        coord = self.get_coordinate(pos)
        if coord:
            row, col = coord
            move = self.board.location_to_move([row, col])
            # Only draw if the move is legal
            if move in self.board.available:
                x = MARGIN + col * GRID_SIZE
                y = MARGIN + row * GRID_SIZE

                current_p = self.board.current_player
                color_rgb = BLACK_STONE_COLOR if current_p == 1 else WHITE_STONE_COLOR

                # Create a surface with per-pixel alpha for transparency
                stone_surface = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                pygame.draw.circle(stone_surface, (*color_rgb, 128), (GRID_SIZE // 2, GRID_SIZE // 2), GRID_SIZE // 2 - 2)
                self.screen.blit(stone_surface, (x - GRID_SIZE // 2, y - GRID_SIZE // 2))

    def draw_info(self):
        """Renders game status information (Turn, Winner, Controls)."""
        if self.game_over:
            if self.winner != -1:
                text = "Black Wins!" if self.winner == 1 else "White Wins!"
                color = (200, 0, 0)
            else:
                text = "Draw Game!"
                color = (50, 50, 200)

            surf = self.big_font.render(text, True, color)
            rect = surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - MARGIN // 2))
            self.screen.blit(surf, rect)

            hint = self.font.render("Press 'R' to Switch Sides", True, (50, 50, 50))
            self.screen.blit(hint, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - MARGIN // 2 + 30))
        else:
            turn_text = "Your Turn" if self.is_human_turn() else "AI Thinking..."
            color = (0, 100, 0) if self.is_human_turn() else (200, 0, 0)
            surf = self.font.render(turn_text, True, color)
            self.screen.blit(surf, (20, SCREEN_HEIGHT - 30))

            side_text = f"You are {'Black' if self.human_plays_black else 'White'}"
            self.screen.blit(self.font.render(side_text, True, (50, 50, 50)), (SCREEN_WIDTH - 200, SCREEN_HEIGHT - 30))

    def is_human_turn(self):
        """Checks if it is currently the human player's turn."""
        if self.human_plays_black:
            return self.board.current_player == 1
        else:
            return self.board.current_player == 2

    def check_end(self):
        """Checks if the game has ended and updates state."""
        end, winner = self.board.game_end()
        if end:
            self.game_over = True
            self.winner = winner
            print(f"Game Over. Winner: {winner}")
            return True
        return False

    def ai_move(self):
        """Executes the AI's move using MCTS."""
        if self.game_over: return

        # Force UI update so "AI Thinking..." is visible
        self.draw_all()
        pygame.display.flip()

        # Get move probabilities from MCTS
        # temp=1e-3 ensures deterministic play (choosing the most visited node)
        acts, probs = self.mcts.get_move_probs(self.board, temp=1e-3)
        move = acts[np.argmax(probs)]

        # Apply move to board and update MCTS tree
        self.board.do_move(move)
        self.mcts.update_with_move(move)

        print(f"AI plays move: {self.board.move_to_location(move)}")
        self.check_end()

    def draw_all(self):
        """Helper to redraw the entire scene."""
        self.draw_board()
        self.draw_stones()
        # Draw hover effect only if it's human turn
        mouse_pos = pygame.mouse.get_pos()
        if self.is_human_turn():
            self.draw_hover(mouse_pos)
        self.draw_info()

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()

        while True:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # Key Controls
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game(switch_side=True)

                # Mouse Controls
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    if self.is_human_turn():
                        coord = self.get_coordinate(event.pos)
                        if coord:
                            row, col = coord
                            move = self.board.location_to_move([row, col])

                            if move in self.board.available:
                                # Human Move
                                self.board.do_move(move)
                                self.mcts.update_with_move(move)
                                print(f"Human plays: {coord}")

                                self.check_end()

            # 2. AI Logic (Triggered outside event loop)
            if not self.game_over and not self.is_human_turn():
                self.ai_move()

            # 3. Rendering
            self.draw_all()
            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    game = GomokuGUI()
    game.run()
