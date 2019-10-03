#! /usr/bin/env python3
import os
import pprint
import random
from collections import defaultdict
from typing import Tuple, List, Any, Dict

import pandas

from env import Mark
from .agent import Agent


class State:
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

    def __add__(self, other):
        total_value: float = self.value + other.value
        total_count: int = self.count + other.count
        return State(total_value, total_count)


class TemporalDifference(Agent):

    def __init__(self, exploratory_rate: float = 0.1, learning_rate: float = 0.5,
                 state_values: Dict[Tuple[Mark], State] = None):
        """
        Represents an agent learning with temporal difference
        :param exploratory_rate: The probability of selecting a random action
        :param learning_rate: How much to learn from the most recent action
        :param state_values: A mapping of states to and their associated values
        """
        self.datafile: str = f"{TemporalDifference.__name__}_{exploratory_rate}_{learning_rate}"
        if not state_values:
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
        import copy

        max_index: int = 0
        max_value: int = 0
        for index, action in enumerate(available_actions):
            next_state: int = copy.deepcopy(state)[action]
            next_value: float = self.state_values[next_state].value

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

        self.previous_state = state
        self.previous_reward = reward

        previous_state_value = self.state_values[self.previous_state]
        current_state_value = self.state_values[state]
        previous_state_value.value += 1 / (previous_state_value.count + 1) * (
                reward + current_state_value.value - previous_state_value.value)

        self.state_values[state].count += 1

    def merge(self, agent):
        """
        Merge state values of another TD agent with this one
        :param agent: another temporal difference agent
        """
        for key, value in agent.state_values.items():
            self.state_values[key] += value

    @staticmethod
    def load_state_values(datafile: str) -> Dict[Tuple[Mark], State]:
        """
        Load data from csv file and convert it to state values
        :param datafile: The name of a csv file containing the data
        :return: The state value mapping
        """
        if os.path.exists(datafile):
            data = pandas.read_csv(datafile)
            data = data.replace({pandas.np.nan: None}).values.tolist()
            state_values = defaultdict(State)
            for d in data:
                state_values[tuple(d[1:11])] = State(value=d[11], count=d[12])
            return state_values
        else:
            return defaultdict(State)

    @staticmethod
    def save_state_values(state_values: Dict[Tuple[Mark], State], datafile: str):
        """
        Save the state values into a csv file
        :param state_values: The state value mapping
        :param datafile: The name of the file to write to
        """
        data: List[Tuple[Any, ...]] = []
        for key, value in state_values.items():
            if isinstance(key, tuple):
                data.append(key + (value.value, value.count))

        df = pandas.DataFrame(data)
        df.to_csv(datafile)
