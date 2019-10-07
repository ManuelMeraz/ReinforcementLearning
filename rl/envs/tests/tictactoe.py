from rl.envs import tictactoe


def test_check_game_status():
    board = ["X"] * 3 + [None] * 6
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = [None] * 3 + ["X"] * 3 + [None] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = [None] * 6 + ["X"] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = ["X", None, None] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = [None, "X", None] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = [None, None, "X"] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.X_WINS

    board = ["O"] * 3 + [None] * 6
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = [None] * 3 + ["O"] * 3 + [None] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = [None] * 6 + ["O"] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = ["O", None, None] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = [None, "O", None] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = [None, None, "O"] * 3
    assert tictactoe.game_status(board) == tictactoe.Status.O_WINS

    board = [None] * 2 + ["O"] + [None] * 6
    assert tictactoe.game_status(board) == tictactoe.Status.IN_PROGRESS

    board = ["O", "X", "O", "O", "X", "X", "X", "O", "O"]
    assert tictactoe.game_status(board) == tictactoe.Status.DRAW


def test_step():
    env = tictactoe.TicTacToeEnv()
    assert env.state == [None] * 9
    assert env.current_player == "X"
    assert env.done is False
    assert env.board_size == 9
    assert env.info["status"].value == tictactoe.Status.IN_PROGRESS.value

    action = 0
    board = ["X"] + [None] * 8
    obs, reward, done, info = env.step(action)
    assert env.state == board
    assert env.current_player == "O"
    assert env.done is False

    action = 1
    board[action] = "O"
    obs, reward, done, info = env.step(action)
    assert env.state == board
    assert env.current_player == "X"
    assert env.done is False
    assert reward == 0

    action = 3
    board[action] = "X"
    obs, reward, done, info = env.step(action)
    assert env.state == board
    assert env.current_player == "O"
    assert env.done is False
    assert reward == 0

    action = 2
    board[action] = "O"
    obs, reward, done, info = env.step(action)
    assert env.state == board
    assert env.current_player == "X"
    assert env.done is False
    assert reward == 0

    action = 6
    board[action] = "X"
    obs, reward, done, info = env.step(action)
    assert env.state == board
    assert env.current_player == "X"
    assert env.done is True
    assert reward == 1

    env.reset()
    assert env.state == [None] * 9
    assert env.current_player == "X"
    assert env.done is False
    assert env.board_size == 9
    assert env.info["status"].value == tictactoe.Status.X_WINS.value
