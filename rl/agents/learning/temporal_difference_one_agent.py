#! /usr/bin/env python3
from typing import Callable

import numpy

from rl.agents.learning import LearningAgent
from rl.reprs import Value


class TemporalDifferenceOne(LearningAgent):
    """
    Applies the temporal difference algorithm as a learning algorithm
    V(s) = V(s) + alpha * (reward + V(s') - V(s))
    Where alpha is the learning rate at 1 / (N + 1)
    """

    def __init__(self, learning_rate: Callable[[int], float], discount_rate: float, *args, **kwargs):
        """
        Represents an agent learning with temporal difference
        :param learning_rate: Either a float or a function that takes in the count (N) of that state and returns 1/N
        """
        super().__init__(*args, **kwargs)
        self.learning_rate: Callable[[int], float] = learning_rate
        self.discount_rate: float = discount_rate

    def learn_value(self):
        """
        Apply temporal difference learning and update the state and values of this agent
        """
        current_transition = self.trajectory[-1]
        current_value: Value = self.state_values[current_transition.state]
        current_value.count += 1

    def reset(self):
        n = len(self.trajectory)
        discount_rates = numpy.zeros(shape=(n, n), dtype='f4')
        rewards = numpy.zeros(n, dtype='f4')
        values = numpy.zeros(n, dtype='f4')

        for i, transition in enumerate(self.trajectory):
            for j in range(0, i):
                discount_rates[i][j] = self.discount_rate ** (n - 1 - j)

            for j in range(i, n):
                discount_rates[i][j] = self.discount_rate ** (n - 1 - i)

            rewards[i] = transition.reward
            values[i] = self.state_values[transition.state].value

        values = discount_rates.dot(rewards) - values
        for i, transition in enumerate(self.trajectory):
            value = self.state_values[transition.state]
            value.value = values[i] / value.count
            self.state_values[transition.state] = value

        self.trajectory.clear()
