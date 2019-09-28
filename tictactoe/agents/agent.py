#! /usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Tuple


class Agent:
    @abstractmethod
    def act(self, state):
        pass

    @staticmethod
    def available_actions(state: Tuple) -> List[int]:
        actions = []
        for action, slot in enumerate(state):
            if slot is None:
                actions.append(action)
        return actions
