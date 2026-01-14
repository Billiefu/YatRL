"""
Copyright (C) 2026 Fu Tszkok

:module: test
:function: Evaluates the performance of four trained agent personalities by running a large number of simulated games, collecting statistics on scoring, doubt rates, and strategic success, and plotting the results.
:author: Fu Tszkok
:date: 2026-01-14
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import torch
import numpy as np
import os
from tqdm import tqdm

from agent import AggressiveBlufferAgent, AggressiveChallengerAgent, ConservativeAgent, RationalAgent
from bar import LiarBarEnv
from utils import GameLogger, plot_scores, plot_strategy_stats, MODEL_DIR, RESULT_DIR

# --- Evaluation Configuration ---
TEST_EPISODES = 1000  # Number of games to simulate for evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM = 33  # Must match the training configuration
HISTORY_DIM = 8  # Must match the training configuration

# --- Agent Setup ---
# Define the agent classes and their corresponding names for loading and reporting
AGENT_CLASSES = [AggressiveBlufferAgent, AggressiveChallengerAgent, ConservativeAgent, RationalAgent]
AGENT_NAMES = [cls.__name__.replace("Agent", "") for cls in AGENT_CLASSES]


def main():
    """Main function to run the evaluation pipeline.
    It loads the trained models, simulates games, collects statistics,
    and generates plots to visualize the results.
    """
    # --- 1. Model and Environment Initialization ---
    print(f"Device: {DEVICE}")
    print("Loading models...")

    agents = [
        AggressiveBlufferAgent(0, STATE_DIM, HISTORY_DIM, device=DEVICE),
        AggressiveChallengerAgent(1, STATE_DIM, HISTORY_DIM, device=DEVICE),
        ConservativeAgent(2, STATE_DIM, HISTORY_DIM, device=DEVICE),
        RationalAgent(3, STATE_DIM, HISTORY_DIM, device=DEVICE)
    ]

    # Load the final trained weights for each agent
    for i, agent in enumerate(agents):
        name = AGENT_NAMES[i]
        path = os.path.join(MODEL_DIR, f"final_{name}.pth")
        if os.path.exists(path):
            agent.policy.load_state_dict(torch.load(path, map_location=DEVICE))
            agent.policy.eval()  # Set the policy to evaluation mode
            print(f"  Loaded {name}")
        else:
            raise FileNotFoundError(f"Model {path} not found. Please run train.py first!")

    # Initialize the environment with the loaded agents and disable console logging
    env = LiarBarEnv(agents, log_callback=None)
    scores_history = [[] for _ in range(4)]  # Stores the final score of each agent per game

    # Initialize a dictionary to store detailed statistics for each agent
    stats = {i: {'turns': 0, 'doubts': 0, 'strat_attempts': 0, 'strat_success': 0} for i in range(4)}

    # --- 2. Evaluation Loop ---
    print(f"\nRunning {TEST_EPISODES} Evaluation Episodes...")
    print(f"Logs will be saved to {os.path.join(RESULT_DIR, 'evaluate_game/')}")

    for ep in tqdm(range(TEST_EPISODES), desc="Evaluating"):
        log_filename = f"evaluate_game/game_{ep + 1}.txt"
        logger = GameLogger(log_filename)
        env.log_callback = logger.log

        obs = env.reset()
        state, history, hand_mask = obs

        logger.log(f"--- Game Start ---")
        from entity import decode_cards
        for i, ag in enumerate(agents):
            logger.log(f"P{i} ({AGENT_NAMES[i]}) Hand: {decode_cards(ag.hand)}")

        elimination_order = []  # Tracks the order in which agents are eliminated

        # Variables to track the context of the last move for statistical purposes
        last_player_idx = None
        last_was_bluff = False

        while True:
            curr_idx = env.current_player_idx
            curr_agent = agents[curr_idx]
            stats[curr_idx]['turns'] += 1

            # Get agent's action without updating gradients
            action_doubt, action_play, _, _, _ = curr_agent.select_action(state, history)

            # --- Check if the current move is a bluff (for stats) ---
            curr_is_bluff = False
            if action_doubt == 0:
                played_indices = [i for i, x in enumerate(action_play) if x == 1]
                played_cards = [curr_agent.hand[i] for i in played_indices if i < len(curr_agent.hand)]
                # A move is a bluff if any played card is not the table card or a wild card
                if any(card != env.table_card and card != 4 for card in played_cards):
                    curr_is_bluff = True

            # --- Update Strategy Statistics ---
            if action_doubt == 1:
                stats[curr_idx]['doubts'] += 1

                if last_player_idx is not None:
                    # Current player's doubt is a strategic attempt. It succeeds if the last player was bluffing.
                    stats[curr_idx]['strat_attempts'] += 1
                    if last_was_bluff:
                        stats[curr_idx]['strat_success'] += 1

                    # Last player's play was also a strategic attempt. It succeeds if they were NOT bluffing and got doubted.
                    stats[last_player_idx]['strat_attempts'] += 1
                    if not last_was_bluff:
                        stats[last_player_idx]['strat_success'] += 1
            else:
                if last_player_idx is not None:
                    # If the current player plays, the last player's strategy is evaluated.
                    # Their attempt succeeds if they bluffed and were NOT doubted.
                    if last_was_bluff:
                        stats[last_player_idx]['strat_attempts'] += 1
                        stats[last_player_idx]['strat_success'] += 1

            # Log the action taken
            if action_doubt == 0:
                played_indices = [i for i, x in enumerate(action_play) if x == 1]
                played_cards = [curr_agent.hand[i] for i in played_indices if i < len(curr_agent.hand)]
                logger.log(f"Turn P{curr_idx}: Played {decode_cards(played_cards)}")
            else:
                logger.log(f"Turn P{curr_idx}: DOUBT!")

            # Take a step in the environment
            next_obs, rewards, done = env.step(action_doubt, action_play)

            # Update context for the next turn
            if action_doubt == 0:
                last_player_idx = curr_idx
                last_was_bluff = curr_is_bluff
            else: # A doubt action resets the context
                last_player_idx = None
                last_was_bluff = False

            # Record eliminations based on the large penalty reward
            for idx, r in rewards.items():
                if r <= -4.0 and idx not in elimination_order:
                    elimination_order.append(idx)

            if done:
                # --- Game End: Assign Scores ---
                winner = env.winner
                logger.log(f"-> Game Over. Winner: P{winner}")

                round_scores = {i: 0 for i in range(4)}
                # Assign scores based on elimination order (0 for first out, 1 for second, etc.)
                for order, p_idx in enumerate(elimination_order):
                    round_scores[p_idx] = order
                if winner is not None:
                    round_scores[winner] = 3  # Winner gets the highest score
                for i in range(4):
                    scores_history[i].append(round_scores[i])
                break

            state, history, hand_mask = next_obs

    # --- 3. Calculate and Display Final Results ---
    doubt_rates = []
    strat_rates = []

    print("\n=== Final Statistics ===")
    print(f"{'Agent':<25} | {'Score':<6} | {'Doubt%':<8} | {'Strat%':<8}")
    print("-" * 55)

    for i in range(4):
        total_turns = max(1, stats[i]['turns'])
        total_strats = max(1, stats[i]['strat_attempts'])

        # Calculate doubt rate and strategy success rate
        d_rate = stats[i]['doubts'] / total_turns
        s_rate = stats[i]['strat_success'] / total_strats
        avg_score = np.mean(scores_history[i])

        doubt_rates.append(d_rate)
        strat_rates.append(s_rate)

        print(f"P{i} {AGENT_NAMES[i]:<21} | {avg_score:.2f}   | {d_rate:.2f}     | {s_rate:.2f}")

    # --- 4. Plot and Save Results ---
    print("\nPlotting results...")
    plot_scores(scores_history, AGENT_NAMES)
    plot_strategy_stats(doubt_rates, strat_rates, AGENT_NAMES)
    print(f"All results saved to {RESULT_DIR}")


if __name__ == "__main__":
    # Script entry point
    main()
