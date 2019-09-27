#!/usr/bin/env python3

from gym_tictactoe import env

from .agent import Agent

import random


class Base(Agent):
    def __init__(self, mark):
        self.mark = mark

    def next_action(self, state, available_actions):
        for action in available_actions:
            board_state, *_ = env.afteraction_state(state, action)
            game_status = env.check_game_status(board_state)

            if game_status > 0:
                if env.tomark(game_status) == self.mark:
                    return action

        return random.choice(available_actions)
