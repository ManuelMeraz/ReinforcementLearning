#! /usr/bin/env python3
from abc import ABC, abstractmethod

import numpy


class Agent(ABC):
    """
    Agent class serves as an interface definition. Every concrete Agent must
    implement these four functions: act, learn, render, and reset.
    """

    @abstractmethod
    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray):
        pass

    @abstractmethod
    def learn(self, state: numpy.ndarray, action: int, reward: float):
        pass

    @abstractmethod
    def merge(self, agent):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def transition_model(self, state: numpy.ndarray, action: int) -> numpy.ndarray:
        """
        State transition model that describes how the environment state changes when the
        agent performs an action depending on the action and the current state.
        :param state: The state of the environment
        :param action: An action available to the agent
        """
        pass

    @abstractmethod
    def value_model(self, state: numpy.ndarray) -> float:
        """
        Map an action to it's value
        :param state: The state of the environment
        :return: The reward received for taking that action
        """
        pass
