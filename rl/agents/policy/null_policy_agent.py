#! /usr/bin/env python3

import numpy

from .policy_agent import PolicyAgent


class NullPolicy(PolicyAgent):
    """
    The null policy agent is an agent that does nothing. This is used for the agent builder when a policy
    agent is not required, and only a learning agent is desired for building an agent.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray):
        """
        A policy for this agent that maps an state to an action
        :param state: The state of the environment
        :param available_actions: A list of available possible actions (positions on the board to mark)
        """
        pass
