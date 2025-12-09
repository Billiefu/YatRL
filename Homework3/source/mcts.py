"""
Copyright (C) 2025 Fu Tszkok

:module: mcts
:function: Implements Monte Carlo Tree Search (MCTS) for the AlphaZero algorithm.
:author: Fu Tszkok
:date: 2025-12-08
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import copy


class TreeNode(object):
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(self, parent, prior_p):
        """Initializes a tree node.
        :param parent: The parent TreeNode object (None for root).
        :param prior_p: The prior probability of selecting this node's action (from policy network).
        """
        self._parent = parent
        self._children = {}  # Map from action to TreeNode
        self._n_visits = 0   # Number of times this node has been visited
        self._Q = 0          # Mean action value
        self._u = 0          # Upper Confidence Bound value
        self._P = prior_p    # Prior probability

    def expand(self, action_priors):
        """Expands the leaf node by adding children for all legal actions.
        :param action_priors: A list of tuples (action, probability).
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Selects the child node with the highest value (Q + U).
        :param c_puct: A constant determining the level of exploration.
        :return: A tuple (action, next_node).
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """Calculates the value of the node using the PUCT formula.
        :param c_puct: A constant determining the level of exploration.
        :return: The calculated value Q + U.
        """
        # Calculate the Upper Confidence Bound (UCB) value
        # U = c_puct * P * sqrt(Parent_Visits) / (1 + Current_Visits)
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """Updates the node's statistics after a playout.
        :param leaf_value: The evaluation value of the leaf node from the current player's perspective.
        """
        self._n_visits += 1
        # Update Q-value using incremental mean: Q_new = Q_old + (v - Q_old) / n
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Recursively updates the node and its ancestors.
        :param leaf_value: The value to propagate up the tree.
        """
        # If there is a parent, propagate the negated value (switching perspective for zero-sum game)
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Checks if the node is a leaf node (has no children).
        :return: True if leaf, False otherwise.
        """
        return self._children == {}

    def is_root(self):
        """Checks if the node is the root of the tree.
        :return: True if root, False otherwise.
        """
        return self._parent is None


class MCTS(object):
    """Implements the Monte Carlo Tree Search algorithm."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        """Initializes the MCTS object.
        :param policy_value_fn: A function taking a board state and returning (action_probs, value).
        :param c_puct: A constant determining the level of exploration.
        :param n_playout: The number of simulations to run per move.
        """
        self._root = TreeNode(None, 1.0)
        self._policy_value_fn = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Executes a single MCTS simulation (Selection, Expansion, Evaluation, Backup).
        :param state: The current board state (will be modified in-place during simulation).
        """
        node = self._root

        # 1. Selection: Traverse down the tree to a leaf node
        while 1:
            if node.is_leaf():
                break
            # Choose the action with the best Q + U value
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 2. Evaluation: Use the policy-value network to evaluate the leaf state
        action_probs, leaf_value = self._policy_value_fn(state)
        end, winner = state.game_end()

        # 3. Expansion: If the game is not over, expand the tree
        if not end:
            node.expand(action_probs)
        else:
            # If game ended, determine the leaf value based on the winner
            if winner == -1:  # Tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # 4. Backup: Update values up the tree
        # Note: We negate leaf_value here because update_recursive expects the value
        # from the *current* node's perspective, but the recursive call flips it immediately.
        # This ensures the parent gets the correct sign.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts and returns the available actions and their probabilities.
        :param state: The current board state.
        :param temp: The temperature parameter controlling exploration (tau).
        :return: A tuple (actions, probabilities).
        """
        # Run n_playout simulations
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # Calculate visit counts for each action at the root
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        # Calculate probabilities using the softmax of visit counts (adjusted by temperature)
        # probs ~ N^(1/temp)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Updates the tree root to the child node corresponding to the move taken.
        :param last_move: The action index of the move played.
        """
        if last_move in self._root._children:
            # Reuse the subtree for the chosen move
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            # If the move was not explored, reset the tree
            self._root = TreeNode(None, 1.0)


def softmax(x):
    """Computes the softmax values for a vector x.
    :param x: Input array.
    :return: Probability distribution array.
    """
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
