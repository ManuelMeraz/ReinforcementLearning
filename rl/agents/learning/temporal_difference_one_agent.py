#! /usr/bin/env python3
import numpy

from rl.agents.learning import LearningAgent
from rl.agents.reprs import Value


class TemporalDifferenceOne(LearningAgent):
    """
    Applies the temporal difference one algorithm as a learning algorithm
    Vt+1(s) = Vt(s) + alpha * (Gt - Vt(s))
    Where alpha is the learning rate and Gt is the sum of accumulated and discounted rewards for the episode.
    Learns at the end of the episode from the trajectory it took after it is reset.
    """

    def __init__(self, learning_rate: float, discount_rate: float, *args, **kwargs):
        """
        Represents an agent learning with temporal difference
        :param learning_rate: Either a float or a function that takes in the count (N) of that state and returns 1/N
        :param discount_rate: The proportion of the future rewards applies to earlier states
        """
        super().__init__(*args, **kwargs)
        self.learning_rate: float = learning_rate
        self.discount_rate: float = discount_rate

    def learn_value(self):
        """
        For this agent keep track of how many times it has visited the latest state
        """
        current_transition = self.trajectory[-1]
        current_value: Value = self.state_values[current_transition.state]
        current_value.count += 1

    def reset(self):
        """
        Reset the trajectory of the agent for this episode and applies the TD(1) learning algorithm.
        """
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
            value.value = self.learning_rate * values[i]
            self.state_values[transition.state] = value

        self.trajectory.clear()
