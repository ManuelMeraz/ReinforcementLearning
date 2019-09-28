#!/usr/bin/env python3
from tictactoe import agents
from tictactoe.env import Status, TicTacToeEnv
from tictactoe.utils import logging_utils


def play():
    env = TicTacToeEnv(show_number=True)
    obs = env.reset()

    players = {
        "X": agents.TemporalDifference(exploratory_rate=0.1,
                                       learning_rate=0.5),
        "O": agents.TemporalDifference(exploratory_rate=0.1,
                                       learning_rate=0.5),
    }

    num_games = 2
    completed_games = 0
    while completed_games < num_games:
        # env.render()
        current_player = players[obs[-1]]
        action = current_player.act(obs)

        prev_obs = obs
        obs, reward, done, info = env.step(action)
        current_player.learn(state=obs, previous_state=prev_obs, reward=reward)

        if done:
            completed_games += 1
            status = info["status"]

            if status == Status.DRAW:
                print("The game was a draw!")
            else:
                winner = {Status.O_WINS: "O", Status.X_WINS: "X"}
                print(f"The winner is {winner[status]}!")

            env.render(human=True)
            obs = env.reset()
            print("Playing new game.")


def main():
    logger = logging_utils.Logger()
    play()


if __name__ == "__main__":
    main()
