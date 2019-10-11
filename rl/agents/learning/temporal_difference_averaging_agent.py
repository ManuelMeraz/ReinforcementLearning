#! /usr/bin/env python3
from collections import defaultdict
from typing import Dict, Tuple, Union

from rl.agents.learning import LearningAgent
from rl.reprs import Value


class TemporalDifferenceAveragingAgent(LearningAgent):
    """
    Applies the temporal difference algorithm as a learning algorithm
    V(s) = V(s) + alpha * (reward + V(s') - V(s))
    Where alpha is the learning rate at 1 / (N + 1)
    """

    def __init__(self, state_values: Dict[Tuple[Union[int, float]], Value] = None,
                 transitions=None):
        """
        Represents an agent learning with temporal difference
        :param learning_rate: How much to learn from the most recent action
        :param state_values: A mapping of states to and their associated values
        """
        super().__init__(transitions=transitions)
        if not state_values:
            self._state_values = defaultdict(Value)
        else:
            self._state_values = state_values

    @property
    def state_values(self) -> Dict[Tuple[Union[int, float]], Value]:
        return self._state_values

    @state_values.setter
    def state_values(self, state_values: Dict[Tuple[Union[int, float]], Value]):
        self._state_values = state_values

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
                        current_value.value - previous_value.value)

                self.state_values[previous_transition.state] = previous_value

                current_transition = previous_transition
                current_value: Value = self.state_values[current_transition.state]

        self.state_values[current_transition.state] = current_value
