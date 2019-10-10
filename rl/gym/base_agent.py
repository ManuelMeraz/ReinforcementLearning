#! /usr/bin/env python3

import numpy

from rl.agents import RandomPolicyAgent


class BaseAgent(RandomPolicyAgent):

    def __init__(self, action_space):
        self.actions = [action for action in range(action_space.n)]

    def available_actions(self, state: numpy.ndarray) -> numpy.ndarray:
        """
        Determines the available actions for the agent given the state
        :param state: A tuple representing the state of the environment
        :return: A list of actions each representing an action available to the agent
        """

        return self.actions
