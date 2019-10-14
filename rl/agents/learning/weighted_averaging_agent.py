#! /usr/bin/env python3

from rl.agents.learning import LearningAgent
from rl.agents.reprs import Value


class WeightedAveraging(LearningAgent):
    """
    This agent learns by applying the sample averaging algorithm. It uses a rolling average to approximate
    a value function.

    Vt+1(s) = Vt(s) + alpha * (Rt - Vt(s))
    Where alpha is some real number between 0 and 1
    """

    def __init__(self, learning_rate, *args, **kwargs):
        """
        Represents an agent learning with temporal difference
        :param state_values: A mapping of states to and their associated values
        """
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate

    def learn_value(self):
        """
        Apply temporal difference learning and update the state and values of this agent
        """

        current_transition = self.trajectory[-1]
        current_value: Value = self.state_values[current_transition.state]
        current_value.value += self.learning_rate * (current_transition.reward - current_value.value)
        self.state_values[current_transition.state] = current_value
