#! /usr/bin/env python3
import logging

# import gym
# from gym import spaces
from enum import Enum


class Status(Enum):
    IN_PROGRESS = 0
    X_WINS = 1
    O_WINS = 2
    DRAW = 3


def check_game_status(board):
    """
    Return game status by current board status.

    :param board: Current board state
    :type board: list

    :returns: Status 
    """

    player_win_status = {'X': Status.X_WINS, 'O': Status.O_WINS}

    for i in range(3):
        if board[i] and len(set(board[i:9:3])) == 1:
            return player_win_status[board[i]]

    for i in range(0,9, 3):
        if board[i] and len(set(board[i:i+3])) == 1:
            return player_win_status[board[i]]

    if board[0] and len(set(board[0:9:4])) == 1:
            return player_win_status[board[0]]

    if board[2] and len(set(board[2:8:2])) == 1:
            return player_win_status[board[2]]

    if None in board:
        return Status.IN_PROGRESS

    return Status.DRAW


# class TicTacToeEnv(gym.Env):
    # def __init__(self, learning_rate=0.5, show_number=False):
        # self.size = 9

        # # Each location on the board represents an action
        # self.action_space = spaces.Discrete(self.size)

        # # Each location on the board is part of the observation space
        # self.observation_space = spaces.Discrete(self.size)
        # self.learning_rate = learning_rate
        # self.start_mark = 'X'

        # # Display numbers on the board for humans
        # self.show_number = show_number

        # # set numpy random seet
        # self.seed()
        # self.reset()

    # def reset(self):
        # self.board = [None] * self.size
        # self.mark = self.start_mark
        # self.done = False
        # return self._get_obs()

    # def step(self, action):
        # """Step environment by action.

        # :param action: The location on the board to mark with an 'X' or 'O'
        # :param type: int

        # :returns:
            # list: Obeservation
            # int: Reward
            # bool: Done
            # dict: Additional information
        # """
        # assert self.action_space.contains(action), f"Action not available in action space:  {action}"

        # if self.done:
            # reward = 0
            # info = None
            # return self._get_obs(), reward, self.done, info

        # reward = 0
        # self.board[action] = self.mark
        # status = check_game_status(self.board)
        # logging.debug("check_game_status board {} mark '{}'"
                      # " status {}".format(self.board, self.mark, status))
        # if status >= 0:
            # self.done = True
            # if status in [1, 2]:
                # # always called by self
                # reward = O_REWARD if self.mark == 'O' else X_REWARD

        # # switch turn
        # self.mark = next_mark(self.mark)
        # return self._get_obs(), reward, self.done, None

    # def _get_obs(self):
        # return tuple(self.board), self.mark

    # def render(self, mode='human', close=False):
        # if close:
            # return
        # if mode == 'human':
            # self._show_board(print)  # NOQA
            # print('')
        # else:
            # self._show_board(logging.info)
            # logging.info('')

    # def show_episode(self, human, episode):
        # self._show_episode(print if human else logging.warning, episode)

    # def _show_episode(self, showfn, episode):
        # showfn("==== Episode {} ====".format(episode))

    # def _show_board(self, showfn):
        # """Draw tictactoe board."""
        # for j in range(0, 9, 3):
            # def mark(i):
                # return tomark(self.board[i]) if not self.show_number or\
                    # self.board[i] != 0 else str(i+1)
            # showfn(LEFT_PAD + '|'.join([mark(i) for i in range(j, j+3)]))
            # if j < 6:
                # showfn(LEFT_PAD + '-----')

    # def show_turn(self, human, mark):
        # self._show_turn(print if human else logging.info, mark)

    # def _show_turn(self, showfn, mark):
        # showfn("{}'s turn.".format(mark))

    # def show_result(self, human, mark, reward):
        # self._show_result(print if human else logging.info, mark, reward)

    # def _show_result(self, showfn, mark, reward):
        # status = check_game_status(self.board)
        # assert status >= 0
        # if status == 0:
            # showfn("==== Finished: Draw ====")
        # else:
            # msg = "Winner is '{}'!".format(tomark(status))
            # showfn("==== Finished: {} ====".format(msg))
        # showfn('')

    # def available_actions(self):
        # return [i for i, c in enumerate(self.board) if c == 0]
