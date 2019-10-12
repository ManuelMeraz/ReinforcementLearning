#! /usr/bin/env python3

import numpy

from .policy_agent import PolicyAgent


class DecayingEGreedy(PolicyAgent):

    def __init__(self, exploratory_rate: float, *args, **kwargs):
        """
        This agent implements a method that picks an action using an egreedy policy
        :param exploratory_rate: The probably of selecting an action at random from a uniform distribution
        """
        super().__init__(*args, **kwargs)
        self.exploratory_rate = exploratory_rate

    def act(self, state: numpy.ndarray, available_actions: numpy.ndarray) -> int:
        """
        Apply this agents policy and select an action
        :param state:  The state of the environment
        :param available_actions: An array of actions available to the agent
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
            action: int = self.greedy_action(state, available_actions)

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
            next_state: numpy.ndarray = self.transition_model(state, action, copy=True)
            next_value: float = self.value_model(next_state, action)

            if next_value > max_value:
                max_index: int = index
                max_value: float = next_value

        return available_actions[max_index]
