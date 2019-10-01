from env import Status
from tictactoe import env


def test_check_game_status():
    board = ["X"] * 3 + [None] * 6
    assert env.check_game_status(board) == env.Status.X_WINS

    board = [None] * 3 + ["X"] * 3 + [None] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = [None] * 6 + ["X"] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = ["X", None, None] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = [None, "X", None] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = [None, None, "X"] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = ["O"] * 3 + [None] * 6
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None] * 3 + ["O"] * 3 + [None] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None] * 6 + ["O"] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = ["O", None, None] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None, "O", None] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None, None, "O"] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None] * 2 + ["O"] + [None] * 6
    assert env.check_game_status(board) == env.Status.IN_PROGRESS

    board = ["O", "X", "O", "O", "X", "X", "X", "O", "O"]
    assert env.check_game_status(board) == env.Status.DRAW


def test_step():
    tictactoe = env.TicTacToeEnv()
    assert tictactoe.board == [None] * 9
    assert tictactoe.mark == "X"
    assert tictactoe.done is False
    assert tictactoe.learning_rate == 0.5
    assert tictactoe.board_size == 9
    assert tictactoe.info["status"].value == Status.IN_PROGRESS.value

    action = 0
    board = ["X"] + [None] * 8
    obs, reward, done, info = tictactoe.step(action)
    assert tictactoe.board == board
    assert tictactoe.mark == "O"
    assert tictactoe.done is False

    action = 1
    board[action] = "O"
    obs, reward, done, info = tictactoe.step(action)
    assert tictactoe.board == board
    assert tictactoe.mark == "X"
    assert tictactoe.done is False
    assert reward == 0

    action = 3
    board[action] = "X"
    obs, reward, done, info = tictactoe.step(action)
    assert tictactoe.board == board
    assert tictactoe.mark == "O"
    assert tictactoe.done is False
    assert reward == 0

    action = 2
    board[action] = "O"
    obs, reward, done, info = tictactoe.step(action)
    assert tictactoe.board == board
    assert tictactoe.mark == "X"
    assert tictactoe.done is False
    assert reward == 0

    action = 6
    board[action] = "X"
    obs, reward, done, info = tictactoe.step(action)
    assert tictactoe.board == board
    assert tictactoe.mark == "X"
    assert tictactoe.done is True
    assert reward == 1

    tictactoe.reset()
    assert tictactoe.board == [None] * 9
    assert tictactoe.mark == "X"
    assert tictactoe.done is False
    assert tictactoe.learning_rate == 0.5
    assert tictactoe.board_size == 9
    assert tictactoe.info["status"].value == Status.X_WINS.value
