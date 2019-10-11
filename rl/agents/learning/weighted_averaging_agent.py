#! /usr/bin/env python3
from typing import Dict, Tuple, Union

from rl.agents.learning import LearningAgent
from rl.reprs import Value


class WeightedAveragingAgent(LearningAgent):
    """
    Applies the temporal difference algorithm as a learning algorithm
    V(s) = V(s) + alpha * (reward + V(s') - V(s))
    Where alpha is the learning rate at 1 / (N + 1)
    """

    def __init__(self, learning_rate, state_values: Dict[Tuple[Union[int, float]], Value] = None, transitions=None):
        """
        Represents an agent learning with temporal difference
        :param state_values: A mapping of states to and their associated values
        """
        super().__init__(state_values=state_values, transitions=transitions)
        self.learning_rate = learning_rate

    def learn_value(self):
        """
        Apply temporal difference learning and update the state and values of this agent
        """

        current_transition = self.trajectory[-1]
        current_value: Value = self.state_values[current_transition.state]
        current_value.value += self.learning_rate * (current_transition.reward - current_value.value)
        self.state_values[current_transition.state] = current_value
