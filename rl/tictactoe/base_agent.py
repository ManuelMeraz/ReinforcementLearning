#! /usr/bin/env python3
from typing import List

import numpy

from rl.agents import RandomPolicyAgent
from rl.agents.learning.null_learning_agent import NullLearningAgent
from rl.envs.tictactoe import Mark


class BaseAgent(RandomPolicyAgent, NullLearningAgent):
    def available_actions(self, state: numpy.ndarray) -> numpy.ndarray:
        """
        Determines the available actions for the agent given the state
        :param state: A tuple representing the state of the environment
        :return: A list of actions each representing an action available to the agent
        """
        actions: List[int] = []
        for action, slot in enumerate(state):
            if slot == Mark.EMPTY:
                actions.append(action)

        return numpy.array(actions)
