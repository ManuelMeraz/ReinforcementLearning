#! /usr/bin/env python3

import numpy

from .policy_agent import PolicyAgent


class DecayingEGreedy(PolicyAgent):
    def __init__(self, exploratory_rate: float, decay_rate: float, *args, **kwargs):
        """
        This agent implements a method that picks an action using an egreedy policy
        :param exploratory_rate: The probably of selecting an action at random from a uniform distribution
        """
        super().__init__(*args, **kwargs)
        self.initial_exploratory_rate = exploratory_rate
        self.exploratory_rate = exploratory_rate
        self.decay_rate = decay_rate
        self.previous_value = 0

    def decay_exploratory_rate(self):
        self.exploratory_rate = self.exploratory_rate * numpy.exp(-1 * self.decay_rate)

    def reset_exploratory_rate(self):
        self.exploratory_rate = self.initial_exploratory_rate

    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray) -> int:
        """
        Apply this agents policy and select an action
        :param state:  The state of the environment
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: an action
        """
        return self.egreedy_policy(state, available_actions)

    def egreedy_policy(self, state: numpy.ndarray, available_actions: numpy.ndarray) -> int:
        """
        Select action based off a probability of exploring or greedy selection of valuable actions
        :param state:  The current state of the board along with the current mark this agent represents
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: A greedy action or random action to explore
        """
        e: float = numpy.random.random()

        if e < self.exploratory_rate:
            action: int = numpy.random.choice(available_actions)
        else:
            action, state = self.greedy_action(state, available_actions)
            value = self.value_model(state)

            if value < self.previous_value:
                self.reset_exploratory_rate()

        return action

    def greedy_action(self, state: numpy.ndarray, available_actions: numpy.ndarray) -> int:
        """
        Select the action with the associated maximum value
        :param state: The current state of the board along with the current mark this agent represents
        :param available_actions: A list of available possible actions (positions on the board to mark)
        :return: The action with the highest value
        """
        max_value: float = float("-inf")
        max_index: int = 0

        for index, action in enumerate(available_actions):
            probabilities: numpy.ndarray
            states: numpy.ndarray
            probabilities, states = self.transition_model(state.copy(), action)

            if probabilities.any():
                transition_index = numpy.random.choice(numpy.arange(len(states)), p=probabilities)
                next_state: numpy.ndarray = states[transition_index]
                next_value: float = self.value_model(next_state)
            else:
                self.reset_exploratory_rate()
                continue

            if next_value > max_value:
                max_index: int = index
                max_value: float = next_value
                max_state = next_state

        return available_actions[max_index], max_state
