#! /usr/bin/env python3

from abc import ABC, abstractmethod


class Agent:
    @abstractmethod
    def act(self, board_state):
        pass

    def available_actions(self, board_state):
        actions = []
        for action, slot in enumerate(board_state):
            if slot is None:
                actions.append(action)
        return actions
