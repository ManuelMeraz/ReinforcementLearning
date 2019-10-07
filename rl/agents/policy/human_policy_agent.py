#!/usr/bin/env python3
import sys
from abc import abstractmethod

import numpy

from rl.agents.policy.policy_agent import PolicyAgent


class HumanPolicyAgent(PolicyAgent):
    def act(self, state: numpy.ndarray) -> int:
        """
        Requests user input to for agent action
        :param state:  A tuple containing the state of environment
        :return: The action selected
        """
        available_actions: numpy.ndarray = self.available_actions(state)

        while True:
            user_input: str = input(self.render_actions(available_actions))

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

    @abstractmethod
    def render_actions(self, actions: numpy.ndarray) -> str:
        """
         Create human readable string to query user for input
        :param actions: An array of action
        """
        pass
