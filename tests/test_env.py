from tictactoe import env

def test_check_game_status():
    board = ['X'] * 3 + [None] * 6
    assert env.check_game_status(board) == env.Status.X_WINS

    board = [None] * 3 + ['X'] * 3 + [None] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = [None] * 6 + ['X'] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = ['X', None, None] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = [None, 'X', None] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = [None, None, 'X'] * 3
    assert env.check_game_status(board) == env.Status.X_WINS

    board = ['O'] * 3 + [None] * 6
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None] * 3 + ['O'] * 3 + [None] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None] * 6 + ['O'] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = ['O', None, None] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None, 'O', None] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None, None, 'O'] * 3
    assert env.check_game_status(board) == env.Status.O_WINS

    board = [None] * 2 + ['O'] + [None] * 6
    assert env.check_game_status(board) == env.Status.IN_PROGRESS

    board = ['O', 'X', 'O', 'O', 'X', 'X', 'X', 'O', 'O']
    assert env.check_game_status(board) == env.Status.DRAW
