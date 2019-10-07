#! /usr/bin/env python3
from abc import abstractmethod
from typing import Dict, Tuple

import numpy

from rl.agents.agent import Agent
from .value import Value


class LearningAgent(Agent):
    """
    The learning agent implements a learning method and is used for purposes of building a state value map
    """

    @property
    @abstractmethod
    def state_values(self) -> Dict[Tuple[int], Value]:
        pass

    @state_values.setter
    @abstractmethod
    def state_values(self, state_values: Dict[Tuple[int], Value]):
        pass

    @property
    @abstractmethod
    def previous_state(self) -> numpy.ndarray:
        pass

    @previous_state.setter
    @abstractmethod
    def previous_state(self, state: numpy.ndarray):
        pass

    @property
    @abstractmethod
    def previous_reward(self) -> float:
        pass

    @previous_reward.setter
    @abstractmethod
    def previous_reward(self, reward: float):
        pass

    @abstractmethod
    def learn(self, state: numpy.ndarray, reward: float):
        pass

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
