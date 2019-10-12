#!/usr/bin/env python3
import sys

import numpy

from rl.agents.policy.policy_agent import PolicyAgent


class Human(PolicyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray) -> int:
        """
        Requests user input to for agent action
        :param state:  A tuple containing the state of environment
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: The action selected
        """
        while True:
            user_input: str = input(f"available actions: {[action + 1 for action in available_actions]}")

            if user_input.startswith("q") or "quit" in user_input:
                print("quitting!")
                sys.exit(0)

            try:
                action = int(user_input) - 1

                if action not in available_actions:
                    raise ValueError
                else:
                    break

            except ValueError:
                print(f"Illegal action: '{user_input}'")

        return action
