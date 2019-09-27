#!/usr/bin/env python3

from tictactoe.utils import logging_utils
import logging
import sys


class Human(object):
    def __init__(self, mark):
        self.mark = mark

    @logging_utils.logged
    def act(self, board_state):
        available_actions = []
        for action, slot in enumerate(board_state):
            if slot is None:
                available_actions.append(action)

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
