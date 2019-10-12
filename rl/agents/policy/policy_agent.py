#! /usr/bin/env python3

from abc import abstractmethod

import numpy

from rl.agents.agent import Agent


class PolicyAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray):
        """
        A policy for this agent that maps an state to an action
        :param state: The state of the environment
        :param available_actions: An array of actions available to the agent
        """
        pass
