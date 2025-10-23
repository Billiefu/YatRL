"""
Copyright (C) 2025 Fu Tszkok

:module: iteration
:function: Implements dynamic programming algorithms for solving Markov Decision Processes.
:author: Fu Tszkok
:date: 2025-10-23
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import random


def value_iteration(env, gamma=0.9, theta=1e-6):
    """Solves the MDP using the Value Iteration algorithm.
    :param env: An instance of the environment, providing states, actions, and transition dynamics.
    :param gamma: The discount factor for future rewards (float).
    :param theta: A small positive number defining the convergence threshold for the value function.
    :return: A tuple (v, policy, history) where v is the optimal value function (dict) and policy is the optimal policy (dict), and history is a list of value history.
    """
    # Initialize the value function and policy arbitrarily.
    v = {state: 0 for state in env.states}
    policy = {state: '' for state in env.states}
    iteration = 0

    # Store the history of the entire value function dictionary
    history = [v.copy()]


    # Loop until the value function converges.
    while True:
        iteration += 1
        # Stores the maximum change in the value function in this iteration.
        delta = 0
        # Use a copy for updates within the loop
        v_old_iter = v.copy()
        # Iterate over every state in the state space.
        for s in env.states:
            # The terminal state has a fixed value of 0 and no actions.
            if s == env.goal_pos:
                continue
            v_old = v_old_iter[s]

            # Calculate the Q-value for each possible action from the current state.
            action_values = {}
            for action in env.action_keys:
                next_s, reward, _ = env.step(s, action)
                q_value = reward + gamma * v_old_iter[next_s]
                action_values[action] = q_value

            # Update the policy and value function based on the best action (greedy update).
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action
            v[s] = action_values[best_action]

            # Check for convergence by tracking the largest change.
            delta = max(delta, abs(v_old - v[s]))

        # Record the value of the start state after each full sweep.
        history.append(v.copy())

        # If the largest change is smaller than the threshold, the value function has converged.
        if delta < theta:
            break

    print(f'Converged after {iteration} iterations.')
    return v, policy, history


def policy_iteration(env, gamma=0.9, theta=1e-6):
    """Solves the MDP using the Policy Iteration algorithm.
    :param env: An instance of the environment.
    :param gamma: The discount factor for future rewards (float).
    :param theta: A small positive number for the convergence threshold in the policy evaluation step.
    :return: A tuple (v, policy, history) where v is the optimal value function (dict) and policy is the optimal policy (dict), and history is a list of value history.
    """
    # Initialize a random policy and a zero value function.
    policy = {s: random.choice(env.action_keys) for s in env.states if s != env.goal_pos}
    policy[env.goal_pos] = ''
    v = {state: 0 for state in env.states}
    iteration = 0

    # List to store the value of the start state.
    history = [v.copy()]

    # Loop until the policy is stable (no longer changes).
    while True:
        iteration += 1

        # --- 1. Policy Evaluation ---
        # Iteratively compute the value function for the current fixed policy.
        while True:
            delta = 0
            for s in env.states:
                if s == env.goal_pos: continue
                v_old = v[s]
                action = policy[s]
                next_s, reward, _ = env.step(s, action)
                # Update the value based on the Bellman equation for a fixed policy.
                v[s] = reward + gamma * v[next_s]
                delta = max(delta, abs(v_old - v[s]))
            # Stop evaluation once the value function converges for this policy.
            if delta < theta:
                break

        # Record value after full policy evaluation.
        history.append(v.copy())

        # --- 2. Policy Improvement ---
        # Check if the policy can be improved by acting greedily with respect to the new value function.
        policy_stable = True
        for s in env.states:
            if s == env.goal_pos: continue
            old_action = policy[s]

            # Find the best action according to the updated value function v.
            action_values = {}
            for action in env.action_keys:
                next_s, reward, _ = env.step(s, action)
                q_value = reward + gamma * v[next_s]
                action_values[action] = q_value
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action

            # If the best action is different from the old action, the policy is not yet stable.
            if old_action != best_action:
                policy_stable = False

        # If the policy did not change for any state, we have found the optimal policy.
        if policy_stable:
            break

    print(f'Converged after {iteration} iterations.')
    return v, policy, history


def truncated_policy_iteration(env, gamma=0.9, j_truncate=5):
    """Solves the MDP using the Truncated Policy Iteration algorithm.
    :param env: An instance of the environment.
    :param gamma: The discount factor for future rewards (float).
    :param j_truncate: The fixed number of iterations for the policy evaluation step.
    :return: A tuple (v, policy, history) where v is the optimal value function (dict) and policy is the optimal policy (dict), and history is a list of value history.
    """
    # Initialize a random policy and a zero value function.
    policy = {s: random.choice(env.action_keys) for s in env.states if s != env.goal_pos}
    policy[env.goal_pos] = ''
    v = {state: 0 for state in env.states}
    iteration = 0

    # List to store history.
    history = [v.copy()]

    # Loop until the policy is stable.
    while True:
        iteration += 1

        # --- 1. Truncated Policy Evaluation ---
        # Run the evaluation step for a fixed number of iterations (`j_truncate`).
        for _ in range(j_truncate):
            v_old = v.copy()  # Use a copy to ensure updates are based on the previous iteration's values.
            for s in env.states:
                if s == env.goal_pos: continue
                action = policy[s]
                next_s, reward, _ = env.step(s, action)
                v[s] = reward + gamma * v_old[next_s]

        # Record value after the truncated evaluation.
        history.append(v.copy())

        # --- 2. Policy Improvement ---
        # Greedily improve the policy based on the partially evaluated value function.
        policy_stable = True
        for s in env.states:
            if s == env.goal_pos: continue
            old_action = policy[s]
            action_values = {}
            for action in env.action_keys:
                next_s, reward, _ = env.step(s, action)
                q_value = reward + gamma * v[next_s]
                action_values[action] = q_value
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False

        # If the policy has not changed, it has converged.
        if policy_stable:
            break

    print(f'Converged after {iteration} iterations.')
    return v, policy, history
