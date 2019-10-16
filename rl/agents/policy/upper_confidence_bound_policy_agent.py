#! /usr/bin/env python3
from collections import defaultdict

import numpy

from .policy_agent import PolicyAgent


class UpperConfidenceBound(PolicyAgent):
    def __init__(self, confidence: float, *args, **kwargs):
        """
        This agent implements a method that picks an action using an egreedy policy
        :param exploratory_rate: The probably of selecting an action at random from a uniform distribution
        """
        super().__init__(*args, **kwargs)
        self.action_counts = defaultdict(lambda: 1)
        self.confidence = confidence
        self.time = 1

    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray) -> int:
        """
        Apply this agents policy and select an action
        :param state:  The state of the environment
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: an action
        """
        action = self.greedy_action(state, available_actions)
        self.action_counts[action] += 1
        self.time += 1
        return action

    def greedy_action(self, state: numpy.ndarray, available_actions: numpy.ndarray) -> int:
        """
        Select the action with the associated maximum value
        :param state: The current state of the board along with the current mark this agent represents
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: The action with the highest value
        """
        max_value: float = float("-inf")
        max_index: int = 0

        for index, action in enumerate(available_actions):
            probabilities: numpy.ndarray
            states: numpy.ndarray
            probabilities, states = self.transition_model(state.copy(), action)

            if probabilities.any():
                transition_index = numpy.random.choice(numpy.arange(len(states)), p=probabilities)
                next_state: numpy.ndarray = states[transition_index]
                confidence_bound = numpy.log(self.time) / self.action_counts[action]
                confidence_bound = self.confidence * numpy.sqrt(confidence_bound)
                next_value: float = self.value_model(next_state) + confidence_bound
            else:
                continue

            if next_value > max_value:
                max_index: int = index
                max_value: float = next_value

        return available_actions[max_index]
