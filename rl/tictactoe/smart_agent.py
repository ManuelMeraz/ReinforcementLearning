#! /usr/bin/env python3
from typing import List, Dict

import numpy

from rl.agents import TemporalDifferenceAgent, EGreedyPolicyAgent
from rl.agents.learning import Value
from rl.envs.tictactoe import Mark


class SmartAgent(TemporalDifferenceAgent, EGreedyPolicyAgent):
    def __init__(self, learning_rate: float, exploratory_rate: float, state_values: Dict[bytes, Value] = None):
        TemporalDifferenceAgent.__init__(self, learning_rate=learning_rate, state_values=state_values)
        EGreedyPolicyAgent.__init__(self, exploratory_rate=exploratory_rate)

    def transition_model(self, state: numpy.ndarray, action: int, copy: bool = False) -> numpy.ndarray:
        if copy:
            next_state = state.copy()
        else:
            next_state = state

        next_state[action] = state[-1]
        return next_state

    def value_model(self, state: numpy.ndarray, action: int) -> float:
        next_state: numpy.ndarray = self.transition_model(state, action, copy=True)
        return self.state_values[next_state.tobytes()].value

    def available_actions(self, state: numpy.ndarray) -> numpy.ndarray:
        """
        Determines the available actions for the agent given the state
        :param state: A tuple representing the state of the environment
        :return: A list of actions each representing an action available to the agent
        """
        actions: List[int] = []
        for action, slot in enumerate(state):
            if slot == Mark.EMPTY:
                actions.append(action)

        return numpy.array(actions)
