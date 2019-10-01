#!/usr/bin/env python3

import random

from .agent import Agent


class Base(Agent):
    def act(self, state):
        available_actions = super().available_actions(state)

        return random.choice(available_actions)
