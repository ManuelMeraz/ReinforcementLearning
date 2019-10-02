#! /usr/bin/env python3
import enum
import logging
import pprint
from typing import Tuple, List, TypeVar

import gym

from tictactoe.utils import logging_utils

Mark = TypeVar('Mark', str, None)


class Status(enum.Enum):
    IN_PROGRESS = 0
    X_WINS = 1
    O_WINS = 2
    DRAW = 3


def check_game_status(board: List[Mark]) -> Status:
    """
    Return game status by current board status.

    :param board: Current board state. ['X', 'O', or None]
    :returns: Status
    """

    player_win_status = {"X": Status.X_WINS, "O": Status.O_WINS}

    for i in range(3):
        if board[i] and len(set(board[i:9:3])) == 1:
            return player_win_status[board[i]]

    for i in range(0, 9, 3):
        if board[i] and len(set(board[i:i + 3])) == 1:
            return player_win_status[board[i]]

    if board[0] and len(set(board[0:9:4])) == 1:
        return player_win_status[board[0]]

    if board[2] and len(set(board[2:8:2])) == 1:
        return player_win_status[board[2]]

    if None in board:
        return Status.IN_PROGRESS

    return Status.DRAW


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Represents an OpenAI Tic Tac Toe environment
        """
        self.board_size: int = 9

        # Each location on the board represents an action
        self.action_space = gym.spaces.Discrete(self.board_size)

        # Each location on the board is part of the observation space
        self.observation_space = gym.spaces.Discrete(self.board_size)
        self.start_mark: str = "X"
        self.status = Status.IN_PROGRESS
        self.info = {"status": self.status}

        # Display numbers on the board for humans
        self.board: Tuple[Mark] = [None] * self.board_size
        self.mark: Mark = self.start_mark
        self.done: bool = False

        # set numpy random seed
        self.seed()

    @logging_utils.logged
    def reset(self) -> Tuple[Mark]:
        """
        Reset the environment to it's initial state.
        :return: The initial state of the environment.
        """
        self.board = [None] * self.board_size
        self.mark = self.start_mark
        self.done = False
        return self.observation

    @logging_utils.logged
    def step(self, action: int) -> Tuple[Tuple[Mark], int, bool, dict]:
        """
        Step environment by action.
        :param action: The location on the board to mark with an 'X' or 'O'
        :returns: Observation, Reward, Done, Info
        """
        assert self.action_space.contains(action), f"Action not available in action space: {action}"

        reward = 0
        self.board[action] = self.mark
        self.status = check_game_status(self.board)
        self.info["status"] = self.status

        if self.status != Status.IN_PROGRESS:
            self.done = True
            reward = 1

        if not self.done:
            self.mark = "X" if self.mark == "O" else "O"

        return self.observation, reward, self.done, self.info

    @property
    def observation(self) -> Tuple[Mark]:
        """
        The state of this environment is the board along with the current player's turn.
        The first 9 elements are the state of the board in row major order, and the last
        element is the current player's turn.
        :return: The state of the game.
        """
        return tuple(self.board) + (self.mark,)

    def render(self, mode=None):
        """
        Draw tictactoe board
        :param mode:  Only human rendering mode is available
        """
        if mode == "human":
            marks = [str(index + 1) if mark is None else mark for index, mark in enumerate(self.board)]
        else:
            marks = [" " if mark is None else mark for mark in self.board]

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
            "current_mark": self.mark,
            "done": self.done,
            "board_state": self.board,
        })

    def __repr__(self):
        return self.__str__()
