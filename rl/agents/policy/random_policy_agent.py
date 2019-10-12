#!/usr/bin/env python3

import numpy

from rl.agents.policy.policy_agent import PolicyAgent


class Random(PolicyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray):
        """
        Uses a uniform random distribution to determine it's action given a state
        TODO: Act according to different distributions
        :param state: The state of the environment
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: a random action
        """
        return numpy.random.choice(available_actions)
