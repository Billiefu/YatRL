"""
Copyright (C) 2026 Fu Tszkok

:module: entity
:function: Defines core game entities and utility functions, including the Gun for Russian Roulette and card mapping.
:author: Fu Tszkok
:date: 2026-01-14
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv-3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import random

# A mapping from integer card representation to human-readable strings.
CARD_MAP = {0: "J", 1: "Q",  2: "K",  3: "A",  4: "Joker"}


def decode_cards(card_list):
    """Converts a list of integer-represented cards to their string names.
    :param card_list: A list of integers representing cards.
    :return: A list of strings corresponding to the card names.
    """
    if card_list is None:
        return []
    return [CARD_MAP.get(c, str(c)) for c in card_list]


class Gun:
    """Implements the Russian Roulette mechanic for the game."""

    def __init__(self, capacity=6):
        """Initializes the gun's cylinder and state.
        :param capacity: The total number of chambers in the gun's cylinder.
        """
        self.capacity = capacity
        self.cylinder = [0] * capacity
        self.current_chamber = 0
        self.bullet_location = 0
        self.reset()

    def reset(self):
        """Resets the gun for a new round.
        This involves clearing the cylinder, randomly placing one bullet, and resetting the chamber position.
        """
        self.cylinder = [0] * self.capacity
        # Randomly place one bullet in one of the chambers
        self.bullet_location = random.randint(0, self.capacity - 1)
        self.cylinder[self.bullet_location] = 1
        self.current_chamber = 0

    def pull_trigger(self):
        """Simulates pulling the trigger, advancing the chamber.
        :return: A boolean, `True` if the chamber contained a bullet (player is dead), `False` otherwise.
        """
        is_dead = (self.cylinder[self.current_chamber] == 1)
        # Advance to the next chamber for the subsequent trigger pull
        self.current_chamber = (self.current_chamber + 1) % self.capacity
        return is_dead

    def get_survival_prob(self):
        """Calculates the probability of surviving the next trigger pull.
        This is based on the assumption that the bullet is in one of the remaining, un-fired chambers.
        :return: A float representing the probability of survival (e.g., 1/6, 1/5, ...).
        """
        remaining_slots = self.capacity - self.current_chamber
        # The probability of being shot is 1 / (number of remaining slots)
        # If there are no remaining slots (which shouldn't happen), survival is certain.
        return 1.0 - (1.0 / remaining_slots) if remaining_slots > 0 else 1.0

    def print_info(self):
        """Prints a visual representation of the gun's state for debugging."""
        visual_chamber = []
        for i in range(self.capacity):
            if i == self.bullet_location:
                # Differentiate between a chamber with a bullet that has been passed and one that is upcoming
                if i < self.current_chamber:
                    visual_chamber.append("[!]")  # Bullet was here, but chamber is spent
                else:
                    visual_chamber.append("[*]")  # Bullet is in an upcoming chamber
            elif i < self.current_chamber:
                visual_chamber.append("[X]")  # Chamber is spent and was empty
            else:
                visual_chamber.append("[ ]")  # Chamber is upcoming and empty

        chamber_str = "".join(visual_chamber)
        print(f"  [GUN] {chamber_str} (Chamber: {self.current_chamber}, Bullet: {self.bullet_location})")
