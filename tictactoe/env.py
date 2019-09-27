#! /usr/bin/env python3
import enum
import gym


class Status(enum.Enum):
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


class TicTacToeEnv(gym.Env):
    def __init__(self, learning_rate=0.5, show_number=False):
        self.size = 9

        # Each location on the board represents an action
        self.action_space = gym.spaces.Discrete(self.size)

        # Each location on the board is part of the observation space
        self.observation_space = gym.spaces.Discrete(self.size)
        self.learning_rate = learning_rate
        self.start_mark = 'X'
        self.info = {}

        # Display numbers on the board for humans
        self.show_number = show_number

        # set numpy random seet
        self.seed()
        self.reset()

    def reset(self):
        self.board = [None] * self.size
        self.mark = self.start_mark
        self.done = False
        return self.observation()

    def step(self, action):
        """Step environment by action.

        :param action: The location on the board to mark with an 'X' or 'O'
        :param type: int

        :returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action), f"Action not available in action space:  {action}"

        reward = 0
        self.board[action] = self.mark
        status = check_game_status(self.board)
        self.mark = 'X' if self.mark == 'O' else 'O'

        if status != Status.IN_PROGRESS:
            self.done = True
            reward = 1

        return self.observation(), reward, self.done, self.info

    def observation(self):
        return tuple(self.board), self.mark

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]
