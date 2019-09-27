#! /usr/bin/env python3

from abc import ABC, abstractmethod

class Agent:

    @abstractmethod
    def act(self, state, available_actions):
        pass
