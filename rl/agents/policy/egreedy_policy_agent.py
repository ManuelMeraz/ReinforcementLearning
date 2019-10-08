#! /usr/bin/env python3
from abc import abstractmethod

import numpy

from rl.agents.policy import PolicyAgent


class EGreedyPolicyAgent(PolicyAgent):

    def __init__(self, exploratory_rate: float):
        """
        This agent implements a method that picks an action using an egreedy policy
        :param exploratory_rate: The probably of selecting an action at random from a uniform distribution
        """
        self.exploratory_rate = exploratory_rate

    @abstractmethod
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
        pass

    @abstractmethod
    def value_model(self, state: numpy.ndarray, action: int) -> float:
        """
        Map an action to it's value
        :param state: The state of the environment
        :param action: An integer representing an action available to the agent
        :return: The reward received for taking that action
        """
        pass

    def act(self, state: numpy.ndarray) -> int:
        """
        Apply this agents policy and select an action
        :param state:  The state of the environment
        :return: an action
        """
        return self.egreedy_policy(state, self.available_actions(state))

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
            next_state: numpy.ndarray = self.transition_model(state, action)
            next_value: float = self.value_model(next_state, action)
            state = self.transition_model(next_state, action, reverse=True)

            if next_value > max_value:
                max_index: int = index
                max_value: float = next_value

        return available_actions[max_index]
