#! /usr/bin/env python3
import pprint
import random
from collections import defaultdict
from typing import Tuple, List

import pandas as pd

from env import Mark
from .agent import Agent


class State:
    def __init__(self, value=0.0, count=0.0):
        self.value = value
        self.count = count

    def __str__(self):
        return pprint.pformat({"value": self.value, "count": self.count})

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        total_value = self.value + other.value
        total_count = self.count + other.count
        return State(total_value, total_count)


class TemporalDifference(Agent):
    def __init__(self, exploratory_rate: float = 0.5, learning_rate: float = 0.5, state_values=defaultdict(State)):
        self.datafile: str = f"{TemporalDifference.__name__}_{exploratory_rate}_{learning_rate}"
        self.state_values = state_values
        self.learning_rate: float = learning_rate
        self.exploratory_rate: float = exploratory_rate

    def act(self, state):
        return self.egreedy_policy(state, super().available_actions(state))

    def egreedy_policy(self, state: Tuple[Mark], available_actions: List[int]) -> int:
        e: float = random.random()

        if e < self.exploratory_rate:
            action: int = random.choice(available_actions)
        else:
            action: int = self.greedy_action(state, available_actions)

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
        self.state_values[previous_state].value += self.learning_rate * (
                reward -
                self.state_values[previous_state].value)

        self.state_values[state].count += 1

    def merge(self, agent):
        """
        Merge state values of another TD agent with this one
        """
        for key, value in agent.state_values.items():
            self.state_values[key] += value

    @staticmethod
    def load_state_values(datafile):
        data = pd.read_csv(datafile)
        data = data.replace({pd.np.nan: None}).values.tolist()
        state_values = defaultdict(State)
        for d in data:
            state_values[tuple(d[1:11])] = State(d[11], d[12])
        return state_values

    @staticmethod
    def save_state_values(state_values, datafile):
        data = []
        for key, value in state_values.items():
            if isinstance(key, tuple):
                data.append(key + (value.value, value.count))

        df = pd.DataFrame(data)
        df.to_csv(datafile)
