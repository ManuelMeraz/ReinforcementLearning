#! /usr/bin/env python3

import random

from tictactoe import env

from .agent import Agent


class TemporalDifference(Agent):
    def __init__(self, mark, exploratory_rate, learning_rate, state_values):
        self.state_values = state_values
        self.mark = mark
        self.learning_rate = learning_rate
        self.exploratory_rate = exploratory_rate

    def act(self, board_state):
        return self.egreedy_policy(board_state,
                                   super().available_actions(board_state))

    def egreedy_policy(self, board_state, available_actions):
        e = random.random()

        if e < self.exploratory_rate:
            action = self.random_action(available_actions)
        else:
            action = self.greedy_action(board_state, available_actions)

        return action

    def random_action(self, available_actions):
        return random.choice(available_actions)

    def greedy_action(self, board_state, available_actions):
        import copy

        max_index = 0
        max_value = 0
        for index, action in enumerate(available_actions):
            next_board_state = copy.deepcopy(board_state)[action]
            next_value = self.state_values[next_board_state].value

            if next_value > max_value:
                max_index = index
                max_value = next_value

        return available_actions[max_index]
