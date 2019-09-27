#! /usr/bin/env python3

from gym_tictactoe import env
from .agent import Agent

class TemporalDifference(object):
    def __init__(self, mark, exploratory_rate, learning_rate, state_values):
        self.state_values = state_values
        self.mark = mark
        self.learning_rate = learning_rate
        self.exploratory_rate = exploratory_rate
        self.episode_rate = 1.0

    def next_action(self, state, available_actions):
        return self.egreedy_policy(state, available_actions)

    def egreedy_policy(self, state, available_actions, values):
        e = random.random()

        if e < self.exploratory_rate * self.episode_rate:
            action = self.random_action(available_actions)
        else:
            action = self.greedy_action(state, available_actions)

        return action

    def random_action(self, available_actions):
        return random.choice(available_actions)

    def greedy_action(self, state, available_actions):
        max_index = 0
        max_value = 0
        for index, action in enumerate(available_actions):
            next_state = after_action_state(state, action)
            next_value = self.get_value(next_state)

            if next_value > max_value:
                max_index = index
                max_value = next_value

        return available_actions[max_index]

    def get_value(self, state):
        if state not in self.state_values:
            board_state = state[0]
            game_status = check_game_status(board_state)
            value = DEFAULT_VALUE

            if game_status > 0:
                value = env.O_REWARD if self.mark == "O" else X_REWARD

            self.state_values[state] += value

        return self.state_values[state]

    def apply_difference(self, state, next_state, reward):
        value = self.get_value(state)
        next_value = self.get_value(next_state)
        difference = next_value - value
        new_value = value + self.learning_rate * difference

        self.state_value[state] = new_value
