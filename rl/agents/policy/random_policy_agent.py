#!/usr/bin/env python3

import numpy

from rl.agents.policy.policy_agent import PolicyAgent


class RandomPolicyAgent(PolicyAgent):
    def act(self, state: numpy.ndarray):
        """
        Uses a uniform random distribution to determine it's action given a state
        :param state: The state of the environment
        :return: a random action
        """
        return numpy.random.choice(self.available_actions(state))
