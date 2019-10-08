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


def game_status(obs: numpy.ndarray) -> Status:
    """
    Determine the status of the game
    :param obs: The state of the board along with the player who just took a turn
    :returns: Status
    """

    player_win_status = {Mark.X: Status.X_WINS, Mark.O: Status.O_WINS}

    for i in range(3):
        if obs[i] and len(set(obs[i:9:3])) == 1:
            return player_win_status[obs[i]]

    for i in range(0, 9, 3):
        if obs[i] and len(set(obs[i:i + 3])) == 1:
            return player_win_status[obs[i]]

    if obs[0] and len(set(obs[0:9:4])) == 1:
        return player_win_status[obs[0]]

    if obs[2] and len(set(obs[2:8:2])) == 1:
        return player_win_status[obs[2]]

    if Mark.EMPTY in obs:
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
        self.start_mark = Mark.O
        self.status = Status.IN_PROGRESS
        self.info = {"status": self.status}

        self.state: numpy.ndarray = numpy.zeros(self.board_size + 1, dtype=numpy.uint8)
        self.current_player: Mark = self.start_mark
        self.done: bool = False

        # set numpy random seed
        self.seed()

    def next_player(self) -> Mark:
        self.current_player = Mark.X if self.current_player == Mark.O else Mark.O
        self.state[-1] = self.current_player
        return self.current_player

    def reset(self) -> numpy.ndarray:
        """
        Reset the environment to it's initial state.
        :return: The initial state of the environment.
        """
        self.state: numpy.ndarray = numpy.zeros(self.board_size + 1, dtype=numpy.uint8)
        self.current_player: Mark = self.start_mark
        self.done: bool = False
        self.game_ticks: int = 0
        return self.observation

    def step(self, action: int) -> Tuple[numpy.ndarray, float, bool, Dict[str, Status]]:
        """
        Step environment by action.
        :param action: The location on the board to mark [1-9]
        :returns: Observation, Reward, Done, Info
        """
        assert self.action_space.contains(action), f"Action not available in action space: {action}"
        self.game_ticks += 1

        self.state[action]: Mark = self.current_player
        self.status: Status = game_status(self.state)
        self.info["status"]: Status = self.status

        reward: float = 0.0
        if self.status != Status.IN_PROGRESS:
            self.done: bool = True

            if self.status == Status.X_WINS:
                reward: float = 1.0 / self.game_ticks
            elif self.status == Status.O_WINS:
                reward: float = 1.0 / (self.game_ticks - 1)

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
        return self.state

    def render(self, mode=None):
        """
        Draw tictactoe board
        :param mode:  Only human rendering mode is available
        """
        mark_to_string = {Mark.X: "X", Mark.O: "O", Mark.EMPTY: " "}
        if mode == "human":
            marks = [str(index + 1) if mark == Mark.EMPTY else mark_to_string[mark] for index, mark in
                     enumerate(self.state)]
        else:
            marks = [mark_to_string[mark] for mark in self.state]

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
