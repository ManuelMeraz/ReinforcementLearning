#! /usr/bin/env python3

import pprint
from typing import Union, Tuple

import numpy


class Transition:
    def __init__(self, state: numpy.ndarray, action: int, reward: float = 0.0):
        """
        Stores the value of a state and how many times the agent has been in this state
        :param state: The state of the environment
        :param reward:  The reward received by being in that state
        """
        self.state: Tuple[Union[int, float]] = tuple(state)
        self.action: int = action
        self.reward: float = reward

    def __str__(self) -> str:
        return pprint.pformat({"state": self.state, "action": self.action, "reward": self.reward})

    def __repr__(self):
        return self.__str__()
