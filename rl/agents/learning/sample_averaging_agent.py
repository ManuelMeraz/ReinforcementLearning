#! /usr/bin/env python3

from rl.agents.learning import LearningAgent
from rl.reprs import Value, Transition


class SampleAveraging(LearningAgent):
    """
    Applies the temporal difference algorithm as a learning algorithm
    V(s) = V(s) + alpha * (reward + V(s') - V(s))
    Where alpha is the learning rate at 1 / (N + 1)
    """

    def __init__(self, *args, **kwargs):
        """
        Represents an agent learning with temporal difference
        :param state_values: A mapping of states to and their associated values
        """
        super().__init__(*args, **kwargs)

    def learn_value(self):
        """
        Apply temporal difference learning and update the state and values of this agent
        """

        current_transition: Transition = self.trajectory[-1]
        current_value: Value = self.state_values[current_transition.state]
        current_value.count += 1
        current_value.value += (1 / current_value.count) * (current_transition.reward - current_value.value)
        self.state_values[current_transition.state] = current_value
