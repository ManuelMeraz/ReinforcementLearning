#! /usr/bin/env python3
import os
import pprint
import random
from collections import defaultdict

import pandas as pd

from .agent import Agent


class State:
    def __init__(self, value=0.0, count=0.0):
        self.value = value
        self.count = count

    def __str__(self):
        return pprint.pformat({"value": self.value, "count": self.count})

    def __repr__(self):
        return self.__str__()


class TemporalDifference(Agent):
    def __init__(self, exploratory_rate=0.5, learning_rate=0.5):
        self.datafile = f"{TemporalDifference.__name__}_{exploratory_rate}_{learning_rate}"

        if os.path.exists(self.datafile):
            self.state_values = self.load_state_values()
        else:
            self.state_values = defaultdict(State)

        self.learning_rate = learning_rate
        self.exploratory_rate = exploratory_rate

    def act(self, state):
        return self.egreedy_policy(state, super().available_actions(state))

    def egreedy_policy(self, state, available_actions):
        e = random.random()

        if e < self.exploratory_rate:
            action = random.choice(available_actions)
        else:
            action = self.greedy_action(state, available_actions)

        return action

    def greedy_action(self, state, available_actions):
        import copy

        max_index = 0
        max_value = 0
        for index, action in enumerate(available_actions):
            next_state = copy.deepcopy(state)[action]
            next_value = self.state_values[next_state].value

            if next_value > max_value:
                max_index = index
                max_value = next_value

        return available_actions[max_index]

    def learn(self, state, previous_state, reward):
        """
        Apply temporal difference learning
        """
        self.state_values[state].value += self.learning_rate * (
                reward -
                self.state_values[previous_state].value)

        self.state_values[state].count += 1

    def load_state_values(self):
        data = pd.read_csv(self.datafile)
        data = data.replace({pd.np.nan: None}).values.tolist()
        self.state_values = defaultdict(State)
        for d in data:
            self.state_values[tuple(d[1:11])] = State(d[11], d[12])
        return self.state_values

    def save_state_values(self):
        data = []
        for key, value in self.state_values.items():
            if isinstance(key, tuple):
                data.append(key + (value.value, value.count))

        df = pd.DataFrame(data)
        df.to_csv(self.datafile)

    def __del__(self):
        self.save_state_values()
