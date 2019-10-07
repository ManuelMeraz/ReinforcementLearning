#! /usr/bin/env python3
import enum
import logging
import pprint
from typing import Dict, Tuple

import gym
import numpy

from rl.utils import logging


class Mark(enum.IntEnum):
    EMPTY = 0
    X = 1
    O = 2


class Status(enum.IntEnum):
    IN_PROGRESS = 0
    X_WINS = 1
    O_WINS = 2
    DRAW = 3


def game_status(observation: numpy.ndarray[Mark]) -> Status:
    """
    Determine the status of the game
    :param observation: The state of the board along with the player who just took a turn
    :returns: Status
    """

    player_win_status = {Mark.X: Status.X_WINS, Mark.O: Status.O_WINS}

    for i in range(3):
        if observation[i] and len(set(observation[i:9:3])) == 1:
            return player_win_status[observation[i]]

    for i in range(0, 9, 3):
        if observation[i] and len(set(observation[i:i + 3])) == 1:
            return player_win_status[observation[i]]

    if observation[0] and len(set(observation[0:9:4])) == 1:
        return player_win_status[observation[0]]

    if observation[2] and len(set(observation[2:8:2])) == 1:
        return player_win_status[observation[2]]

    if None in observation:
        return Status.IN_PROGRESS

    return Status.DRAW


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Represents an OpenAI Tic Tac Toe environment
        """
        self.board_size: int = 9
        self.game_ticks = 0

        # Each location on the board represents an action
        self.action_space = gym.spaces.Discrete(self.board_size)

        # Each location on the board is part of the observation space
        # The last item is the current player's turn
        self.observation_space = gym.spaces.Discrete(self.board_size + 1)
        self.start_mark = Mark.X
        self.status = Status.IN_PROGRESS
        self.info = {"status": self.status}

        self.state: numpy.ndarray[Mark] = numpy.zeros(self.board_size + 1, dtype=numpy.uint8)
        self.current_player: Mark = self.start_mark
        self.done: bool = False

        # set numpy random seed
        self.seed()

    @logging.logged
    def reset(self) -> numpy.ndarray[Mark]:
        """
        Reset the environment to it's initial state.
        :return: The initial state of the environment.
        """
        self.state: numpy.ndarray = numpy.zeros(self.board_size + 1, dtype=numpy.uint8)
        self.current_player: Mark = self.start_mark
        self.done: bool = False
        self.game_ticks: int = 0
        return self.observation

    @logging.logged
    def step(self, action: int) -> Tuple[numpy.ndarray[Mark], float, bool, Dict[str, Status]]:
        """
        Step environment by action.
        :param action: The location on the board to mark [1-9]
        :returns: Observation, Reward, Done, Info
        """
        assert self.action_space.contains(action), f"Action not available in action space: {action}"
        self.game_ticks += 1

        reward: float = 0.0
        self.state[action]: Mark = self.current_player
        self.status: Status = game_status(self.state)
        self.info["status"]: Status = self.status

        if self.status != Status.IN_PROGRESS:
            self.done: bool = True

            if self.status == Status.X_WINS or self.status == Status.O_WINS:
                reward: float = 1.0 / self.game_ticks

        return self.observation, reward, self.done, self.info

    @property
    def observation(self):
        """
        The state of this environment is the board along with the current player's turn.
        The first 9 elements are the state of the board in row major order, and the last
        element is the current player's turn.
        :return: The state of the game.
        """
        self.state[-1] = self.current_player
        self.current_player = "X" if self.current_player == "O" else "O"
        return self.state

    def render(self, mode=None):
        """
        Draw tictactoe board
        :param mode:  Only human rendering mode is available
        """
        if mode == "human":
            marks = [str(index + 1) if mark is None else mark for index, mark in enumerate(self.state)]
        else:
            marks = [" " if mark is None else mark for mark in self.state]

        board_string = ""
        for j in range(0, 9, 3):
            board_string += "  " + "|".join([marks[i] for i in range(j, j + 3)])
            board_string += "\n"
            if j < 6:
                board_string += "  " + "-----\n"

        if mode == 'human':
            print(board_string)
        else:
            logging.info(pprint.pformat(board_string))

    def __str__(self):
        return pprint.pformat({
            "board_size": 9,
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "current_player": self.current_player,
            "done": self.done,
            "board_state": self.state,
        })

    def __repr__(self):
        return self.__str__()
