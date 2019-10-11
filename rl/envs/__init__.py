from gym.envs.registration import register

register(
    id='TicTacToe-v0',
    entry_point='rl.envs.tictactoe:TicTacToeEnv',
)