#! /usr/bin/env python3
from collections import defaultdict
from typing import Dict, Tuple

import numpy

from rl.agents.learning import LearningAgent, Value


class TemporalDifferenceAgent(LearningAgent):
    """
    Applies the temporal difference algorithm as a learning algorithm
    V(s) = V(s) + alpha * (reward + V(s') - V(s))
    Where alpha is the learning rate at 1 / (N + 1)
    """

    def __init__(self, learning_rate: float, state_values: Dict[Tuple[int], Value] = None):
        """
        Represents an agent learning with temporal difference
        :param learning_rate: How much to learn from the most recent action
        :param state_values: A mapping of states to and their associated values
        """
        if not state_values:
            self._state_values = defaultdict(Value)
        else:
            self._state_values = state_values

        self.learning_rate: float = learning_rate

        self._previous_state: numpy.ndarray = None
        self._previous_reward: float = None

    @property
    def state_values(self) -> Dict[Tuple[int], Value]:
        return self._state_values

    @state_values.setter
    def state_values(self, state_values: Dict[Tuple[int], Value]):
        self._state_values = state_values

    @property
    def previous_state(self) -> numpy.ndarray:
        return self._previous_state

    @previous_state.setter
    def previous_state(self, state: numpy.ndarray):
        self._previous_state = state

    @property
    def previous_reward(self) -> float:
        return self._previous_reward

    @previous_reward.setter
    def previous_reward(self, reward: float):
        self._previous_reward = reward

    def learn(self, state: numpy.ndarray, reward: float):
        """
        Apply temporal difference learning and update the state and values of this agent
        :param state: The current state of the board along with the current mark this agent represents
        :param reward: The reward having taken the most recent action
        """

        if self.previous_state is None and self._previous_reward is None:
            self.state_values[tuple(state)].count += 1
            self.previous_state = state.copy()
            self._previous_reward = reward
            return

        previous_value: Value = self.state_values[tuple(self.previous_state)]
        current_value: Value = self.state_values[tuple(state)]

        current_value.count += 1
        current_value.value += reward

        previous_value.value += 1 / (previous_value.count + 1) * (
                current_value.value - previous_value.value)
        # previous_value.value += self.learning_rate * (
        #         reward + current_value.value - previous_value.value)

        self.state_values[tuple(state)] = current_value
        self.state_values[tuple(self.previous_state)] = previous_value

        self.previous_state = state.copy()
        self._previous_reward = reward
