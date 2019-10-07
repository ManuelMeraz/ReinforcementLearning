#! /usr/bin/env python3
from typing import List

import numpy

from rl.agents import HumanPolicyAgent
from rl.envs.tictactoe import Mark


class HumanAgent(HumanPolicyAgent):
    def available_actions(self, state: numpy.ndarray) -> numpy.ndarray:
        """
        Determines the available actions for the agent given the state
        :param state: A tuple representing the state of the environment
        :return: A list of actions each representing an action available to the agent
        """
        actions: List[numpy.uint8] = []
        for action, slot in enumerate(state):
            if slot == Mark.EMPTY:
                actions.append(action)

        return numpy.array(actions)

    def render_actions(self, actions: numpy.ndarray) -> str:
        """
        Format available actions to a readable version
        :param actions: An array of available actions to the user
        :return: a string to be printed out for a user to read and select an action
        """
        return f"Select a slot: {str([action + 1 for action in actions])}: "
