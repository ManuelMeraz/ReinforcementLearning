#! /usr/bin/env python3
from typing import Dict, Tuple, Union

import numpy

from rl.agents import EGreedyPolicyAgent
from rl.agents.learning import WeightedAveragingAgent
from rl.reprs import Value


class EGreedyWeightedAveraging(EGreedyPolicyAgent, WeightedAveragingAgent):
    def __init__(self, action_space, exploratory_rate: float, learning_rate: float,
                 state_values: Dict[Tuple[Union[int, float]], Value] = None, transitions=None):
        EGreedyPolicyAgent.__init__(self, exploratory_rate=exploratory_rate)
        WeightedAveragingAgent.__init__(self, learning_rate, state_values=state_values, transitions=transitions)

        self.action_space = action_space
        self.actions = numpy.array([action for action in range(action_space.n)])

    def transition_model(self, state: numpy.ndarray, action: int, copy: bool = False) -> numpy.ndarray:
        """
        State transition model that describes how the environment state changes when the
        agent performs an action depending on the action and the current state.
        :param state: The state of the environment
        :param action: An action available to the agent
        :param copy: When applying the action to the state, do so with a copy or apply it directly
        """
        return numpy.array([action])

    def value_model(self, state: numpy.ndarray, action: int) -> float:
        """
        Map an action to it's value
        :param state: The state of the environment
        :param action: An integer representing an action available to the agent
        :return: The reward received for taking that action
        """
        return self.state_values[tuple(state)].value

    def available_actions(self, state: numpy.ndarray) -> numpy.ndarray:
        """
        Determines the available actions for the agent given the state
        :param state: A tuple representing the state of the environment
        :return: A list of actions each representing an action available to the agent
        """
        return self.actions
