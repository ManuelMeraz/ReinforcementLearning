#! /usr/bin/env python3

from rl.agents.learning import LearningAgent
from rl.reprs import Value


class TemporalDifferenceAveraging(LearningAgent):
    """
    Applies the temporal difference algorithm as a learning algorithm
    V(s) = V(s) + alpha * (reward + gamma * V(s') - V(s))
    Where alpha is the learning rate at 1 / (N + 1)
    Gamma is the discount rate
    """

    def __init__(self, discount_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount_rate = discount_rate

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

                previous_value.value += 1 / (previous_value.count + 1) * (
                        self.discount_rate * current_value.value - previous_value.value)

                self.state_values[previous_transition.state] = previous_value

                current_transition = previous_transition
                current_value: Value = self.state_values[current_transition.state]

        self.state_values[current_transition.state] = current_value
