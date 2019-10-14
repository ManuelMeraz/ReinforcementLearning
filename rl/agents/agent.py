#! /usr/bin/env python3
from abc import ABC, abstractmethod

import numpy


class Agent(ABC):
    """
    Agent class serves as an interface definition. Every concrete Agent must
    implement these four functions: act, learn, render, and reset.
    """

    @abstractmethod
    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray) -> int:
        """
        Given a state and a set of actions, the agent selects what action should be taken based off of it's policy
        :param state: The state of the environment (could include the agent)
        :param available_actions: An array of integers, one of which the agent will select and return
        :returns: The action select from available_actions
        """
        pass

    @abstractmethod
    def learn(self, state: numpy.ndarray, action: int, reward: float):
        """
        A learning agent learns from the data passed in
        :param state: The state that the agent just transitioned from
        :param action: The action the agent took
        :param reward: The reward the agent received after committing to an action from a state
        """
        pass

    @abstractmethod
    def merge(self, agent):
        """
        Merge a learning agent into this agent. Combine state-value mappings and learn state-action knowledge.
        :param agent: Any learning agent.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the trajectory of the agent for this episode.
        """
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
