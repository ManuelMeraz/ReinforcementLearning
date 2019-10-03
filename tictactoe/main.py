#!/usr/bin/env python3
import argparse
import math
import multiprocessing
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


def merge_agents(agents: TemporalDifference) -> TemporalDifference:
    main_agent = TemporalDifference()
    for agent in agents:
        main_agent.merge(agent)

    return main_agent


def learn_from_game(args):
    td_agent = args[0]
    num_games = args[1]
    index = args[2]
    num_cpus = args[3]

    for _ in tqdm(range(num_games), desc=f"agent: {index}", total=num_games, position=index % num_cpus):
        env = TicTacToeEnv()
        obs: Tuple[Mark] = env.reset()

        players: Dict[str, TemporalDifference] = {
            "X": TemporalDifference(td_agent.exploratory_rate, td_agent.learning_rate, td_agent.state_values),
            "O": TemporalDifference(td_agent.exploratory_rate, td_agent.learning_rate, td_agent.state_values),
        }

        while True:
            current_player: TemporalDifference = players[obs[-1]]
            action: int = current_player.act(obs)

            obs: Tuple[Mark]
            reward: int
            done: bool
            info: dict

            obs, reward, done, info = env.step(action)
            current_player.learn(state=obs, reward=reward)

            if done:

                if info["status"] == Status.X_WINS:
                    td_agent.merge(players["X"])
                elif info["status"] == Status.O_WINS:
                    td_agent.merge(players["O"])

                break

    return td_agent


def learn(main_agent: TemporalDifference, num_games: int, num_agents: int):
    """
    Pit agents against themselves tournament style. The winners survive.
    :param main_agent: The agent that will learn from playing games
    :param num_games:  The number of games to play each other
    :param num_agents:  The number of games to play each other
    """

    print(f"Learning! Number of games: {num_games} Number of agents: {num_agents}")

    processes = multiprocessing.cpu_count()

    if num_agents < processes:
        processes = num_agents

    chunksize = math.floor(num_agents / processes)
    with multiprocessing.Pool(processes=processes) as pool:
        agents = [(TemporalDifference(main_agent.exploratory_rate, main_agent.learning_rate, main_agent.state_values),
                   num_games, i, processes) for i in range(num_agents)]

        print("Playing games...")
        agents = pool.map(learn_from_game, iterable=agents, chunksize=chunksize)

        print("Merging knowledge...")
        for agent in agents:
            main_agent.merge(agent)

    TemporalDifference.save_state_values(main_agent.state_values, main_agent.datafile)


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

        play(player_X=agentTypes[suboptions.X], player_O=agentTypes[suboptions.O])

    elif options.command == "learn":
        subparser = subparsers.add_parser("learn",
                                          help="Pit two temporal agents against each other and generate a value map.")
        subparser.add_argument("-n", "--num-games",
                               help="The number of games to play against each other.", default=100)
        subparser.add_argument("-a", "--num-agents",
                               help="The number of agents to pit against themselves.", default=1000)
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               default=0.1)
        subparser.add_argument("-l", "--learning-rate",
                               help="The amount to scale the learning of the temporal difference algorithm.",
                               default=0.5)
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])

        agent = TemporalDifference(exploratory_rate=suboptions.exploratory_rate, learning_rate=suboptions.learning_rate)
        learn(agent, num_games=int(suboptions.num_games), num_agents=int(suboptions.num_agents))
    else:
        print("No other options.")


if __name__ == "__main__":
    main()
