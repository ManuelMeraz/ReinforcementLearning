#! /usr/bin/env python3
import os
import pprint
import random
from collections import defaultdict
from typing import Tuple, List, Dict

import pandas

from env import Mark
from .agent import Agent


class Value:
    def __init__(self, value: float = 0.0, count: int = 0):
        """
        Stores the value of a state and how many times the agent has been in this state
        :param value: The total accumulated reward computed using temporal difference
        :param count:  The number of times the agent has been in this state
        """
        self.value = value
        self.count = count

    def __str__(self) -> str:
        return pprint.pformat({"value": self.value, "count": self.count})

    def __repr__(self):
        return self.__str__()


class TemporalDifference(Agent):

    def __init__(self, exploratory_rate: float = 0.0, learning_rate: float = 0.5,
                 state_values: Dict[Tuple[Mark], Value] = None, load_data: bool = True):
        """
        Represents an agent learning with temporal difference
        :param exploratory_rate: The probability of selecting a random action
        :param learning_rate: How much to learn from the most recent action
        :param state_values: A mapping of states to and their associated values
        """
        self.datafile: str = f"{TemporalDifference.__name__}_{exploratory_rate}_{learning_rate}"
        if not state_values and not load_data:
            self.state_values = defaultdict(Value)
        elif not state_values:
            self.state_values = TemporalDifference.load_state_values(self.datafile)
        else:
            self.state_values = state_values

        self.learning_rate: float = learning_rate
        self.exploratory_rate: float = exploratory_rate
        self.previous_state: Tuple[Mark] = None
        self.previous_reward: int = None

    def act(self, state: Tuple[Mark]) -> int:
        """
        Apply this agents policy and select an action
        :param state:  The current state of the board along with the current mark this agent represents
        :return: an action
        """
        return self.egreedy_policy(state, super().available_actions(state))

    def egreedy_policy(self, state: Tuple[Mark], available_actions: List[int]) -> int:
        """
        Select action based off a probability of exploring or greedy selection of valuable actions
        :param state:  The current state of the board along with the current mark this agent represents
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: A greedy action or random action to explore
        """
        e: float = random.random()

        if e < self.exploratory_rate:
            action: int = random.choice(available_actions)
        else:
            action: int = self.greedy_action(state, available_actions)

        return action

    def greedy_action(self, state: Tuple[Mark], available_actions: List[int]) -> int:
        """
        Select the action with the associated maximum value
        :param state: The current state of the board along with the current mark this agent represents
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: The action with the highest value
        """

        current_player_mark = state[-1]
        max_value: int = 0
        # max_index: int = random.choice(range(len(available_actions)))
        max_index: int = 0
        for index, action in enumerate(available_actions):
            next_state: List[Mark] = list((*state[:9], "X" if state[-1] == "O" else "O"))
            next_state[action] = current_player_mark
            next_value: float = self.state_values[tuple(next_state)].value

            if next_value > max_value:
                max_index: int = index
                max_value: int = next_value

        return available_actions[max_index]

    def learn(self, state: Tuple[Mark], reward: int):
        """
        Apply temporal difference learning and update the state and values of this agent
        :param state: The current state of the board along with the current mark this agent represents
        :param reward: The reward having taken the most recent action
        """

        if self.previous_state is None and self.previous_reward is None:
            self.previous_state = state
            self.previous_reward = reward
            return

        previous_value = self.state_values[self.previous_state]
        current_value = self.state_values[state]

        previous_value.value += 1 / (previous_value.count + 1) * (
                reward + current_value.value - previous_value.value)
        # previous_value.value += self.learning_rate * (
        #         reward + current_value.value - previous_value.value)

        self.state_values[state].count += 1
        self.state_values[self.previous_state] = previous_value

        self.previous_state = state
        self.previous_reward = reward

    def merge(self, agent):
        """
        Merge state values of another TD agent with this one
        :param agent: another temporal difference agent
        """
        for state, other_value in agent.state_values.items():
            if other_value.value == 0:
                continue

            value = self.state_values[state];
            if value.count == 0 and value.value == 0:
                value = other_value
            # else:
            #     value.value = (value.value + other_value.value) / 2
                # total_count = value.count + other_value.count
                # value.value = value.count * value.value + other_value.count * other_value.value
                # value.value /= total_count

            # value.count = 0
            self.state_values[state] = value

    @staticmethod
    def load_state_values(datafile: str) -> Dict[Tuple[Mark], Value]:
        """
        Load data from csv file and convert it to state values
        :param datafile: The name of a csv file containing the data
        :return: The state value mapping
        """
        if os.path.exists(datafile):
            data = pandas.read_csv(datafile)
            data = data.replace({pandas.np.nan: None}).values.tolist()
            state_values = defaultdict(Value)
            for d in data:
                state_values[tuple(d[1:11])] = Value(value=d[11], count=d[12])
            return state_values
        else:
            return defaultdict(Value)

    @staticmethod
    def save_state_values(state_values: Dict[Tuple[Mark], Value], datafile: str):
        """
        Save the state values into a csv file
        :param state_values: The state value mapping
        :param datafile: The name of the file to write to
        """
        data = []
        for key, value in state_values.items():
            if isinstance(key, tuple):
                data.append(key + (value.value, value.count))

        df = pandas.DataFrame(data)
        df.to_csv(datafile)
