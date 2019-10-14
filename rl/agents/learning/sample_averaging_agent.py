#! /usr/bin/env python3

from rl.agents.learning import LearningAgent
from rl.agents.reprs import Value, Transition


class SampleAveraging(LearningAgent):
    """
    This agent learns by applying the sample averaging algorithm. It uses a rolling average to approximate
    a value function.

    Vt+1(s) = Vt(s) + alpha * (Rt - Vt(s))
    Where alpha is 1/N
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn_value(self):
        """
        Apply sample averaging learning and update the state and values of this agent
        """
        current_transition: Transition = self.trajectory[-1]
        current_value: Value = self.state_values[current_transition.state]
        current_value.count += 1
        current_value.value += (1 / current_value.count) * (current_transition.reward - current_value.value)
        self.state_values[current_transition.state] = current_value
