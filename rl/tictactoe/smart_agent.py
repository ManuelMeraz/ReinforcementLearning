#! /usr/bin/env python3
from typing import Dict, Tuple, Union

import numpy

from rl.agents import TemporalDifferenceAgent, EGreedyPolicyAgent
from rl.envs.tictactoe import Mark
from rl.reprs import Value


class SmartAgent(TemporalDifferenceAgent, EGreedyPolicyAgent):
    def __init__(self, learning_rate: float, exploratory_rate: float,
                 state_values: Dict[Tuple[Union[int, float]], Value] = None, transitions=None):
        TemporalDifferenceAgent.__init__(self, learning_rate=learning_rate, state_values=state_values,
                                         transitions=transitions)
        EGreedyPolicyAgent.__init__(self, exploratory_rate=exploratory_rate)

    def transition_model(self, state: numpy.ndarray, action: int, copy: bool = False,
                         reverse: bool = False) -> numpy.ndarray:
        """
        State transition model that describes how the environment state changes when the
        agent performs an action depending on the action and the current state.
        :param state: The state of the environment
        :param action: An action available to the agent
        :param copy: When applying the action to the state, do so with a copy or apply it directly
        :param reverse: Reverse the action passed in
        """
        if copy:
            next_state = state.copy()
        else:
            next_state = state

        if reverse:
            next_state[action] = Mark.EMPTY
        else:
            next_state[action] = state[-1]
        state_counts = self.transitions[(*next_state, action)]

        if not state_counts:
            next_state[action] = state[-1]
            return next_state

        states = list(state_counts.keys())
        counts = numpy.array(list(state_counts.values()))

        sum = counts.sum()
        counts = counts / sum
        index = numpy.random.choice(numpy.arange(len(states)), p=counts)
        return numpy.array(states[index])

    def value_model(self, state: numpy.ndarray, action: int) -> float:
        """
        Map an action to it's value
        :param state: The state of the environment
        :param action: An integer representing an action available to the agent
        :return: The reward received for taking that action
        """
        return self.state_values[tuple(state)].value

    def available_actions(self, state: numpy.ndarray) -> numpy.ndarray:
        """
        Determines the available actions for the agent given the state
        :param state: A tuple representing the state of the environment
        :return: A list of actions each representing an action available to the agent
        """
        return numpy.where(state == Mark.EMPTY)[0]
