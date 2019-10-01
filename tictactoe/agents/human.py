#!/usr/bin/env python3
import sys
from typing import List, Tuple

from env import Mark
from .agent import Agent


class Human(Agent):
    def act(self, state: Tuple[Mark]) -> int:
        """
        Requests user input to place a mark on the board
        :param state:  A tuple containing the state of the board and the player's mark
        :return:
        """
        available_actions: List[int] = super().available_actions(state)

        while True:
            user_input: str = input("Enter move[1-9]: ")

            if user_input.startswith("q") or "quit" in user_input:
                print("quitting the game!")
                sys.exit(0)

            try:
                action = int(user_input) - 1

                if action not in available_actions:
                    raise ValueError
                else:
                    break

            except ValueError:
                print(f"Illegal location: '{user_input}'")

        return action
