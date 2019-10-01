#! /usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Tuple


class Agent(ABC):
    @abstractmethod
    def act(self, state):
        pass

    @staticmethod
    def available_actions(state: Tuple) -> List[int]:
        """
        Determines the available actions for the agent given the state
        :param state: A tuple representing the state of the environment
        :return: A list of actions each representing an action available to the agent
        """
        actions: List[int] = []
        for action, slot in enumerate(state):
            if slot is None:
                actions.append(action)
        return actions
