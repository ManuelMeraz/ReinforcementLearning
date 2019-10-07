#! /usr/bin/env python3

from abc import abstractmethod

import numpy

from rl.agents.agent import Agent


class PolicyAgent(Agent):
    @abstractmethod
    def act(self, state):
        """
        A policy for this agent that maps an state to an action
        :param state: The state of the environment
        """
        pass

    @abstractmethod
    def available_actions(self, state: numpy.ndarray) -> numpy.ndarray:
        """
        Given a state, determine the available actions
        :param state: The state of the environment
        """
        pass
