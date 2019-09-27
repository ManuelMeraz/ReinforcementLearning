#!/usr/bin/env python3

from tictactoe.utils import logging_utils
from .agent import Agent
import logging
import sys


class Human(Agent):
    def act(self, board_state):
        available_actions = super().available_actions(board_state)

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
