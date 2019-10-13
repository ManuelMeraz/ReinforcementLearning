#! /usr/bin/env python3
from typing import Callable, Union

from rl.agents.learning import LearningAgent
from rl.reprs import Value


class TemporalDifference(LearningAgent):
    """
    Applies the temporal difference algorithm as a learning algorithm
    V(s) = V(s) + alpha * (reward + V(s') - V(s))
    Where alpha is the learning rate at 1 / (N + 1)
    """

    def __init__(self, learning_rate: Union[float, Callable[[int], float]], discount_rate: float, *args, **kwargs):
        """
        Represents an agent learning with temporal difference
        :param learning_rate: Either a float or a function that takes in the count (N) of that state and returns 1/N
        """
        super().__init__(*args, **kwargs)
        if isinstance(learning_rate, float):
            learning_rate = lambda n: learning_rate

        self.learning_rate: Callable[[int], float] = learning_rate
        self.discount_rate: float = discount_rate

    def learn_value(self):
        """
        Apply temporal difference learning and update the state and values of this agent
        """
        current_transition = self.trajectory[-1]
        current_value: Value = self.state_values[current_transition.state]
        current_value.count += 1
        current_value.value += current_transition.reward

        if current_value.value != 0:
            for i in range(-2, -1 * len(self.trajectory), -1):
                previous_transition = self.trajectory[i - 1]
                previous_value: Value = self.state_values[previous_transition.state]

                previous_value.value += self.learning_rate(previous_value.count + 1) * (
                        self.discount_rate * current_value.value - previous_value.value)

                self.state_values[previous_transition.state] = previous_value

                current_transition = previous_transition
                current_value: Value = self.state_values[current_transition.state]

        self.state_values[current_transition.state] = current_value
