#! /usr/bin/env python3
from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Agent class serves as an interface definition. Every concrete Agent must
    implement these four functions: act, learn, render, and reset.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def act(self, **kwargs):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass
