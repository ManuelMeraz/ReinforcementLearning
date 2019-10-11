#! /usr/bin/env python3

import numpy

from rl.agents.policy import PolicyAgent


class NullPolicyAgent(PolicyAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    def act(self, state: numpy.ndarray):
        """
        A policy for this agent that maps an state to an action
        :param state: The state of the environment
        """
        pass

    def available_actions(self, state: numpy.ndarray) -> numpy.ndarray:
        """
        Given a state, determine the available actions
        :param state: The state of the environment
        """
        pass
