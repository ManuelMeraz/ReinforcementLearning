import numpy

from rl.envs import tictactoe
from rl.envs.tictactoe import Mark, Status


def test_check_game_status():
    board = numpy.array([Mark.X] * 3 + [Mark.EMPTY] * 6)
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = numpy.array([Mark.EMPTY] * 3 + [Mark.X] * 3 + [Mark.EMPTY] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = numpy.array([Mark.EMPTY] * 6 + [Mark.X] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = numpy.array([Mark.X, Mark.EMPTY, Mark.EMPTY] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = numpy.array([Mark.EMPTY, Mark.X, Mark.EMPTY] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = numpy.array([Mark.EMPTY, Mark.EMPTY, Mark.X] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = numpy.array([Mark.O] * 3 + [Mark.EMPTY] * 6)
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = numpy.array([Mark.EMPTY] * 3 + [Mark.O] * 3 + [Mark.EMPTY] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = numpy.array([Mark.EMPTY] * 6 + [Mark.O] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = numpy.array([Mark.O, Mark.EMPTY, Mark.EMPTY] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = numpy.array([Mark.EMPTY, Mark.O, Mark.EMPTY] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = numpy.array([Mark.EMPTY, Mark.EMPTY, Mark.O] * 3)
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = numpy.array([Mark.EMPTY] * 2 + [Mark.O] + [Mark.EMPTY] * 6)
    assert tictactoe.game_status(board) == tictactoe.Status.IN_PROGRESS

    board = numpy.array([Mark.O, Mark.X, Mark.O, Mark.O, Mark.X, Mark.X, Mark.X, Mark.O, Mark.O])
    assert tictactoe.game_status(board) == tictactoe.Status.DRAW


def test_step():
    env = tictactoe.TicTacToeEnv()
    assert numpy.array_equal(env.state, numpy.array([Mark.EMPTY] * 9))
    assert env.player == Mark.X
    assert env.done is False
    assert env.board_size == 9
    assert env.info["status"].value == Status.IN_PROGRESS.value

    action = 0
    board = numpy.array([Mark.X] + [Mark.EMPTY] * 8)
    obs, reward, done, info = env.step(action)
    assert numpy.array_equal(env.state, board)
    assert env.player == Mark.O
    assert env.done is False

    action = 1
    board[action] = Mark.O
    obs, reward, done, info = env.step(action)
    assert numpy.array_equal(env.state, board)
    assert env.player == Mark.X
    assert env.done is False
    assert reward == 0

    action = 3
    board[action] = Mark.X
    obs, reward, done, info = env.step(action)
    assert numpy.array_equal(env.state, board)
    assert env.player == Mark.O
    assert env.done is False
    assert reward == 0

    action = 2
    board[action] = Mark.O
    obs, reward, done, info = env.step(action)
    assert numpy.array_equal(env.state, board)
    assert env.player == Mark.X
    assert env.done is False
    assert reward == 0

    action = 6
    board[action] = Mark.X
    obs, reward, done, info = env.step(action)
    assert numpy.array_equal(env.state, board)
    assert env.player == Mark.O
    assert env.done is True
    assert reward == 1 / 5

    env.reset()
    assert numpy.array_equal(env.state, numpy.array([Mark.EMPTY] * 9))
    assert env.player == Mark.X
    assert env.done is False
    assert env.board_size == 9
    assert env.info["status"].value == Status.X_WINS.value
