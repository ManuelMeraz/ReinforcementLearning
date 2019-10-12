#! /usr/bin/env python3
from abc import abstractmethod
from collections import defaultdict, Counter

import numpy

from rl.agents.agent import Agent
from rl.reprs import Transition
from rl.reprs.value import Value


class LearningAgent(Agent):
    """
    The learning agent implements a learning method and is used for purposes of building a state value map
    """

    def __init__(self, state_values=None, transitions=None):
        self.trajectory = []

        if state_values is None:
            self.state_values = defaultdict(Value)
        else:
            self.state_values = state_values

        if transitions is None:
            self.transitions = defaultdict(Counter)
        else:
            self.transitions = transitions

    @abstractmethod
    def learn_value(self):
        pass

    def reset(self):
        self.trajectory.clear()

    def learn(self, transition: Transition):
        if not self.trajectory:
            self.state_values[transition.state].count += 1
            self.trajectory.append(transition)
            return

        self.trajectory.append(transition)
        self.learn_value()
        self.learn_transition()

    def learn_transition(self):
        transition = self.trajectory.pop()
        previous_transition = self.trajectory.pop()

        state_action_pair = (*previous_transition.state, previous_transition.action)
        transition_counts = self.transitions[state_action_pair]
        transition_counts[transition.state] += 1

        self.trajectory.append(previous_transition)
        self.trajectory.append(transition)

    def merge(self, agent):
        """
        Merge state values of another learning agent agent with this one
        :param agent: another learning agent
        """
        assert isinstance(agent, LearningAgent), "Agent being merged must be a learning agent"

        for state, other_value in agent.state_values.items():
            value: Value = self.state_values[state]

            if not value.count:
                value = other_value
            else:
                # value.value = (value.value + other_value.value) / 2
                total_count = value.count + other_value.count
                value.value = value.count * value.value + other_value.count * other_value.value
                value.value /= total_count
                value.count = total_count / 2

            self.state_values[state] = value

        for state_action, counts in agent.transitions.items():
            self.transitions[state_action] += counts

    def transition_model(self, state: numpy.ndarray, action: int, copy: bool = False) -> numpy.ndarray:
        """
        State transition model that describes how the environment state changes when the
        agent performs an action depending on the action and the current state.
        :param state: The state of the environment
        :param action: An action available to the agent
        :param copy: When applying the action to the state, do so with a copy or apply it directly
        """
        if copy:
            next_state = state.copy()
        else:
            next_state = state

        state_counts = self.transitions[(*next_state, action)]

        if not state_counts:
            return state

        states = list(state_counts.keys())
        counts = numpy.array(list(state_counts.values()))

        counts = numpy.maximum(counts, 0)
        sum = counts.sum()
        probabilities = counts / sum

        # values = []
        # for p, s in zip(probabilities, states):
        #     values.append(self.state_values[s].value)

        index = numpy.random.choice(numpy.arange(len(state_counts)), p=probabilities)
        # return states[numpy.argmax(numpy.array(values))]
        return states[index]

    def value_model(self, state: numpy.ndarray, action: int) -> float:
        """
        Map an action to it's value
        :param state: The state of the environment
        :param action: An integer representing an action available to the agent
        :return: The reward received for taking that action
        """
        return self.state_values[tuple(state)].value
