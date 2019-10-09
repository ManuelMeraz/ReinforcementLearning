#! /usr/bin/env python3
from abc import abstractmethod
from collections import defaultdict, Counter
from typing import Dict, Tuple, Union

from rl.agents.agent import Agent
from rl.reprs import Transition
from rl.reprs.value import Value


class LearningAgent(Agent):
    """
    The learning agent implements a learning method and is used for purposes of building a state value map
    """

    def __init__(self, transitions=None):
        self.previous_transition = None

        if transitions is None:
            self.transitions = defaultdict(Counter)
        else:
            self.transitions = transitions

    @property
    @abstractmethod
    def state_values(self) -> Dict[Tuple[Union[int, float]], Value]:
        pass

    @state_values.setter
    @abstractmethod
    def state_values(self, state_values: Dict[Tuple[Union[int, float]], Value]):
        pass

    @abstractmethod
    def learn_value(self, state: Tuple[Union[int, float]], reward: float):
        pass

    def learn(self, transition: Transition):
        if self.previous_transition is None:
            self.previous_transition = transition
            self.state_values[transition.state].count += 1
            return

        self.learn_value(transition.state, transition.reward)
        self.learn_transition(transition)
        self.previous_transition = transition

    def learn_transition(self, transition: Transition):
        state_action_pair = (*self.previous_transition.state, self.previous_transition.action)
        transition_counts = self.transitions[state_action_pair]
        transition_counts[transition.state] += 1

    def merge(self, agent):
        """
        Merge state values of another learning agent agent with this one
        :param agent: another learning agent
        """
        assert isinstance(agent, LearningAgent), "Agent being merged must be a learning agent"

        for state, other_value in agent.state_values.items():
            value: Value = self.state_values[state]

            if not value.count:
                value = other_value
            else:
                # value.value = (value.value + other_value.value) / 2
                total_count = value.count + other_value.count
                value.value = value.count * value.value + other_value.count * other_value.value
                value.value /= total_count
                value.count = total_count / 2

            self.state_values[state] = value

        for state_action, counts in agent.transitions.items():
            self.transitions[state_action] += counts
