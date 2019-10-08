import pprint

import numpy


class Trajectory:
    def __init__(self, state: numpy.ndarray = 0.0, reward: float = 0.0):
        """
        Stores the value of a state and how many times the agent has been in this state
        :param state: The state of the environment
        :param reward:  The reward received by being in that state
        """
        self.state = tuple(state)
        self.reward = reward

    def __str__(self) -> str:
        return pprint.pformat({"state": self.value, "reward": self.reward})

    def __repr__(self):
        return self.__str__()
