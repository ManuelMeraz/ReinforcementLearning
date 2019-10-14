#! /usr/bin/env python3

from rl.agents.learning import LearningAgent
from rl.agents.reprs import Value


class TemporalDifferenceZeroAveraging(LearningAgent):
    """
    Applies the temporal difference zero (TD(0)) algorithm as a learning algorithm
    Vt+1(s) = Vt(s) + alpha * (reward + gamma * Vt(s') - Vt(s))
    Where alpha is the learning rate at 1/N and gamma is the discount rate.
    """

    def __init__(self, discount_rate: float, *args, **kwargs):
        """
        Represents an agent learning with temporal difference
        :param learning_rate: Either a float or a function that takes in the count (N) of that state and returns 1/N
        """
        super().__init__(*args, **kwargs)
        self.discount_rate: float = discount_rate

    def learn_value(self):
        """
        Apply temporal difference zero learning and update the state and values of this agent
        """
        current_transition = self.trajectory[-1]
        current_value: Value = self.state_values[current_transition.state]
        current_value.count += 1
        current_value.value += current_transition.reward

        if current_value.value != 0:
            previous_transition = self.trajectory[-2]
            previous_value: Value = self.state_values[previous_transition.state]

            previous_value.value += (1 / previous_value.count) * (
                    self.discount_rate * current_value.value - previous_value.value)

            self.state_values[previous_transition.state] = previous_value

        self.state_values[current_transition.state] = current_value
