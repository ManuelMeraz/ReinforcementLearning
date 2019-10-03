#!/usr/bin/env python3
import argparse
import math
import multiprocessing
import os
import signal
import sys
from collections import defaultdict
from typing import Dict, Tuple, Union

from tqdm import tqdm

from agents import Human, Base, TemporalDifference
from agents.temporal_difference import State
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


def merge_agents(agents: TemporalDifference) -> TemporalDifference:
    main_agent = TemporalDifference()
    for agent in agents:
        main_agent.merge(agent)

    return main_agent


def learn_from_game(td_agent):
    env = TicTacToeEnv()
    obs: Tuple[Mark] = env.reset()

    players: Dict[str, TemporalDifference] = {
        "X": td_agent,
        "O": td_agent,
    }

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
            break

    return td_agent


def learn(num_episodes: int = 100, learning_rate: float = 0.5, exploratory_rate: float = 0.1):
    """
    Pit two temporal difference agents against each other to learn the value of each state
    :param exploratory_rate: The probability of sampling an action from a uniform random distribution
                            instead of selecting the greedy action.
    :param learning_rate:  The amount to learn from previous observations
    :param num_episodes:  The number of games to play each other
    """

    print(f"Learning! Number of episodes: {num_episodes}")

    datafile = f"TemporalDifference_{exploratory_rate}_{learning_rate}"
    if os.path.exists(datafile):
        state_values = TemporalDifference.load_state_values(datafile)
    else:
        state_values = defaultdict(State)

    processes = multiprocessing.cpu_count()
    chunksize = math.floor(num_episodes / processes)
    with multiprocessing.Pool(processes=processes) as pool:
        agents = (TemporalDifference(exploratory_rate, learning_rate, state_values) for x in range(num_episodes))
        print("Playing games.")
        agents = list(tqdm(pool.imap(learn_from_game, iterable=agents, chunksize=chunksize), total=num_episodes))

        print("Merging knowledge.")
        size = int(math.ceil(float(len(agents)) / processes))
        agents = [agents[i * size:(i + 1) * size] for i in range(processes)]

        agents = list(tqdm(pool.imap(merge_agents, iterable=agents), total=len(agents)))

        main_agent = TemporalDifference(exploratory_rate, learning_rate)
        for agent in agents:
            main_agent.merge(agent)

        state_values = main_agent.state_values

    TemporalDifference.save_state_values(state_values, datafile)


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    parser = argparse.ArgumentParser();
    parser.add_argument("command", help="Play a game of TicTacToe or Train the TemporalDifference agent.")

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

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
                      "td": TemporalDifference(exploratory_rate=0.1, learning_rate=0.5)}

        agentTypes[suboptions.X].state_values = TemporalDifference.load_state_values(agentTypes[suboptions.X].datafile)
        play(player_X=agentTypes[suboptions.X], player_O=agentTypes[suboptions.O])

    elif options.command == "learn":
        subparser = subparsers.add_parser("learn",
                                          help="Pit two temporal agents against each other and generate a value map.")
        subparser.add_argument("-n", "--num-episodes",
                               help="The number of episodes (games) to play against each other.", default=10000)
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               default=0.1)
        subparser.add_argument("-l", "--learning-rate",
                               help="The amount to scale the learning of the temporal difference algorithm.",
                               default=0.5)
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])
        learn(num_episodes=int(suboptions.num_episodes), exploratory_rate=suboptions.exploratory_rate,
              learning_rate=suboptions.learning_rate)
    else:
        print("No other options.")


if __name__ == "__main__":
    main()
