#!/usr/bin/env python3

from tictactoe import env
from .agent import Agent

import random


class Base(Agent):
    def __init__(self, mark):
        self.mark = mark

    def act(self, board_state):
        available_actions = super().available_actions(board_state)

        return random.choice(available_actions)
