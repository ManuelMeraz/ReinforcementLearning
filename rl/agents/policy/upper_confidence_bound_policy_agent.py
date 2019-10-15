#! /usr/bin/env python3
import sys
from collections import defaultdict

import numpy

from .policy_agent import PolicyAgent


class UpperConfidenceBound(PolicyAgent):
    def __init__(self, confidence: float, decay_rate: float, *args, **kwargs):
        """
        This agent implements a method that picks an action using an egreedy policy
        :param exploratory_rate: The probably of selecting an action at random from a uniform distribution
        """
        super().__init__(*args, **kwargs)
        self.upper_bounds = defaultdict(lambda: sys.float_info.max)
        self.action_counts = defaultdict(lambda: 1)
        self.decay_rate = decay_rate
        self.confidence = confidence

    def decay(self, bound, average_value):
        new_bound = bound * numpy.exp(-1 * self.decay_rate) + 1
        if new_bound < average_value:
            new_bound = sys.float_info.max

        return new_bound

    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray) -> int:
        """
        Apply this agents policy and select an action
        :param state:  The state of the environment
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: an action
        """
        action, average_value = self.greedy_action(state, available_actions)
        self.action_counts[action] += 1

        self.upper_bounds[action] = self.decay(self.upper_bounds[action], average_value)
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

        average_value = 0
        for index, action in enumerate(available_actions):
            probabilities: numpy.ndarray
            states: numpy.ndarray
            probabilities, states = self.transition_model(state.copy(), action)

            if probabilities.any():
                transition_index = numpy.random.choice(numpy.arange(len(states)), p=probabilities)
                next_state: numpy.ndarray = states[transition_index]
                confidence_bound = numpy.log(self.upper_bounds[action]) / self.action_counts[action]
                confidence_bound = self.confidence * numpy.sqrt(confidence_bound)
                next_value: float = self.value_model(next_state)
                average_value += next_value
                next_value += confidence_bound
            else:
                continue

            if next_value > max_value:
                max_index: int = index
                max_value: float = next_value

        average_value /= len(available_actions)
        return available_actions[max_index], average_value
