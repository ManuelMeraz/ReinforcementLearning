#!/usr/bin/env python3
from typing import Dict, Tuple

from agents import Human
from tictactoe import agents
from tictactoe.env import TicTacToeEnv, Status, Mark
from utils.logging_utils import Logger


def play():
    env: TicTacToeEnv = TicTacToeEnv(show_number=True)
    obs: tuple = env.reset()

    players: Dict[str, Human] = {
        "X": agents.Human(),
        "O": agents.Human(),
    }

    num_games: int = 10
    completed_games: int = 0
    while completed_games < num_games:
        env.render(mode='human')
        current_player: Human = players[obs[-1]]
        action: int = current_player.act(obs)

        prev_obs = obs

        obs: Tuple[Mark]
        reward: int
        done: bool
        info: dict
        obs, reward, done, info = env.step(action)
        # current_player.learn(state=obs, previous_state=prev_obs, reward=reward)

        if done:
            completed_games += 1
            status: Status = info["status"]

            if status == Status.DRAW:
                print("The game was a draw!")
            else:
                winner = {Status.O_WINS: "O", Status.X_WINS: "X"}
                print(f"The winner is {winner[status]}!")

            env.render(mode='human')
            obs = env.reset()
            print("Playing new game.")


def main():
    logger: Logger = Logger()
    play()


if __name__ == "__main__":
    main()
