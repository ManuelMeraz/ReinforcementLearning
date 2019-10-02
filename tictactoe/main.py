#!/usr/bin/env python3
import argparse
import signal
import sys
from typing import Dict, Tuple, Union

from tqdm import tqdm

from agents import Human, Base, TemporalDifference
from tictactoe.env import TicTacToeEnv, Status, Mark
from utils.logging_utils import Logger


def play(player_X: Union[Human, Base, TemporalDifference], player_O: Union[Human, Base, TemporalDifference]):
    """
    Play game of TicTactoe
    :param player_X: Player X
    :param player_O:  Player O
    """
    env = TicTacToeEnv()
    obs: Tuple[Mark] = env.reset()

    players: Dict[str, Union[Human, Base, TemporalDifference]] = {
        "X": player_X,
        "O": player_O,
    }

    mode = 'human' if isinstance(player_X, Human) or isinstance(player_O, Human) else None

    while True:
        env.render(mode=mode)
        current_player: Union[Human, Base, TemporalDifference] = players[obs[-1]]
        action: int = current_player.act(obs)

        obs: Tuple[Mark]
        reward: int
        done: bool
        info: dict
        obs, reward, done, info = env.step(action)

        if done:
            status: Status = info["status"]

            if mode == "human":
                if status == Status.DRAW:
                    print("The game was a draw!")
                else:
                    winner = {Status.O_WINS: "O", Status.X_WINS: "X"}
                    print(f"The winner is {winner[status]}!")

                print("Playing new game.")

            env.render(mode=mode)
            obs = env.reset()


def learn(num_episodes: int = 100, learning_rate: float = 0.5, exploratory_rate: float = 0.5):
    """
    Pit two temporal difference agents against each other to learn the value of each state
    :param exploratory_rate: The probability of sampling an action from a uniform random distribution
                            instead of selecting the greedy action.
    :param learning_rate:  The amount to learn from previous observations
    :param num_episodes:  The number of games to play each other
    """
    env = TicTacToeEnv()
    obs: Tuple[Mark] = env.reset()

    players: Dict[str, TemporalDifference] = {
        "X": TemporalDifference(exploratory_rate=exploratory_rate, learning_rate=learning_rate),
        "O": TemporalDifference(exploratory_rate=exploratory_rate, learning_rate=learning_rate),
    }

    current_episode = 0
    print(f"Learning! Number of episodes: {num_episodes}")
    for _ in tqdm(range(num_episodes)):
        while True:
            current_player: TemporalDifference = players[obs[-1]]
            action: int = current_player.act(obs)

            prev_obs: Tuple[Mark] = obs
            obs: Tuple[Mark]
            reward: int
            done: bool
            info: dict
            obs, reward, done, info = env.step(action)
            current_player.learn(state=obs, previous_state=prev_obs, reward=reward)

            if done:
                current_episode += 1
                env.render()
                obs = env.reset()
                break


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    parser = argparse.ArgumentParser();
    parser.add_argument("command", help="Play a game of TicTacToe or Train the TemporalDifference agent.")
    options = parser.parse_args(sys.argv[1:2])
    subparsers = parser.add_subparsers()

    if options.command == "play":
        subparser = subparsers.add_parser("play", help="Play a game of TicTacToe!")
        subparser.add_argument("-X", choices=["human", "base", "td"], help="Human, Base, or Temporal Difference",
                               default="td")
        subparser.add_argument("-O", choices=["human", "base", "td"], help="Human, Base, or Temporal Difference",
                               default="human")
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])
        agentTypes = {"human": Human(), "base": Base(),
                      "td": TemporalDifference(exploratory_rate=0.5, learning_rate=0.5)}
        play(player_X=agentTypes[suboptions.X], player_O=agentTypes[suboptions.O])

    elif options.command == "learn":
        subparser = subparsers.add_parser("learn",
                                          help="Pit two temporal agents against each other and generate a value map.")
        subparser.add_argument("-n", "--num-episodes",
                               help="The number of episodes (games) to play against each other.", default=10000)
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               default=0.5)
        subparser.add_argument("-l", "--learning-rate",
                               help="The amount to scale the learning of the temporal difference algorithm.",
                               default=0.5)
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])
        learn(num_episodes=suboptions.num_episodes, exploratory_rate=suboptions.exploratory_rate,
              learning_rate=suboptions.learning_rate)
    else:
        print("No other options.")


if __name__ == "__main__":
    main()
