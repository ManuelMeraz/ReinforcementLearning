#!/usr/bin/env python3

from tictactoe import env
from .agent import Agent

import random


class Base(Agent):
    def act(self, board_state):
        available_actions = available_actions(board_state)

        return random.choice(available_actions)
