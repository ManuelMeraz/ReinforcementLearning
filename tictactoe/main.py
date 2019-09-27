#!/usr/bin/env python3
from collections import defaultdict

from tictactoe import agents
from tictactoe.env import Status, TicTacToeEnv
from tictactoe.utils import logging_utils


class State:
    def __init__(self):
        self.value = 0
        self.count = 0


state_values = defaultdict(State)


def play():
    env = TicTacToeEnv(show_number=True)
    obs = env.reset()

    players = {
        "X":
        agents.Human("X"),
        "O":
        agents.TemporalDifference("O",
                                  exploratory_rate=0.1,
                                  learning_rate=0.5,
                                  state_values=state_values),
    }

    while True:
        env.render(human=True)

        action = players[obs["current_turn"]].act(obs["board"])
        obs, reward, done, info = env.step(action)

        if done:
            status = obs["status"]

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
