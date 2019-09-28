#!/usr/bin/env python3
from typing import List

import sys

from .agent import Agent


class Human(Agent):
    def act(self, board_state):
        """

        :param board_state:  A row major ordered list representation of the board 
        :type board_state: A list of integers
        :return:
        """
        available_actions: List[int] = super().available_actions(board_state)

        while True:
            user_input = input("Enter move[1-9]: ")

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
