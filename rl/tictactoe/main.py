#!/usr/bin/env python3
import argparse
import math
import multiprocessing
import os
import signal
import sys
import time
from typing import Dict

import gym
import numpy
from tqdm import tqdm

from rl.agents import AgentBuilder, Agent
from rl.envs.tictactoe import Status, Mark
from rl.utils.io import load_learning_agent, save_learning_agent
from rl.utils.logging import Logger


def next_player(current_player) -> Mark:
    return Mark.X if current_player == Mark.O else Mark.O


def available_actions(state: numpy.ndarray) -> numpy.ndarray:
    return numpy.where(state == Mark.EMPTY)[0]


def play(player_x: Agent, player_o: Agent):
    """
    Play game of TicTactoe
    :param player_x: Player X
    :param player_o:  Player O
    """
    env = gym.make("TicTacToe-v0")
    obs: numpy.ndarray = env.reset()

    players: Dict[Mark, Agent] = {
        Mark.X: player_x,
        Mark.O: player_o,
    }

    observations = {
        Mark.X: numpy.append(obs, Mark.X),
        Mark.O: numpy.append(obs, Mark.O)
    }

    env.render(mode="human")
    while True:
        previous_player = env.next_player()
        current_player = env.current_player()
        action: int = players[current_player].act(observations[current_player], available_actions(obs))

        obs: numpy.ndarray
        reward: float
        done: bool
        info: Dict[str, Status]
        obs, reward, done, info = env.step(action)
        players[current_player].learn(state=observations[current_player], action=action, reward=reward)
        players[previous_player].learn(state=observations[previous_player], action=action, reward=-1 * reward)

        observations[Mark.X] = numpy.append(obs, Mark.X)
        observations[Mark.O] = numpy.append(obs, Mark.O)

        env.render(mode="human")

        if done:
            if info["status"] == Status.X_WINS:
                print(f"The winner is X!")
                players[Mark.X].learn(observations[Mark.X], action, reward)
                players[Mark.O].learn(observations[Mark.O], action, -1 * reward)
            elif info["status"] == Status.O_WINS:
                print(f"The winner is O!")
                players[Mark.O].learn(observations[Mark.O], action, reward)
                players[Mark.X].learn(observations[Mark.X], action, -1 * reward)
            else:
                print("The game was a draw!")
                players[Mark.O].learn(observations[Mark.O], action, reward)
                players[Mark.X].learn(observations[Mark.X], action, reward)

            print("Playing new game.")
            players[Mark.O].reset()
            players[Mark.X].reset()
            obs = env.reset()
            env.render(mode="human")


def learn_from_game(args):
    builder = args[0]
    num_games = args[1]
    index = args[2]
    num_cpus = args[3]

    td_agent = builder.make()
    env = gym.make("TicTacToe-v0")
    obs: numpy.ndarray = env.reset()

    players: Dict[Mark, Agent] = {
        Mark.X: builder.make(),
        Mark.O: builder.make(),
    }

    observations = {
        Mark.X: numpy.append(obs, Mark.X),
        Mark.O: numpy.append(obs, Mark.O)
    }

    for _ in tqdm(range(num_games), desc=f"agent: {index}", total=num_games, position=index % num_cpus):
        while True:
            previous_player = env.next_player()
            current_player = env.current_player()
            action: int = players[current_player].act(observations[current_player], available_actions(obs))

            obs: numpy.ndarray
            reward: float
            done: bool
            info: Dict[str, Status]
            obs, reward, done, info = env.step(action)
            players[current_player].learn(state=observations[current_player], action=action, reward=reward)
            players[previous_player].learn(state=observations[previous_player], action=action, reward=-1 * reward)

            observations[Mark.X] = numpy.append(obs, Mark.X)
            observations[Mark.O] = numpy.append(obs, Mark.O)

            if done:
                if info["status"] == Status.X_WINS:
                    players[Mark.X].learn(observations[Mark.X], action, reward)
                    players[Mark.O].learn(observations[Mark.O], action, -1 * reward)
                elif info["status"] == Status.O_WINS:
                    players[Mark.O].learn(observations[Mark.O], action, reward)
                    players[Mark.X].learn(observations[Mark.X], action, -1 * reward)
                else:
                    players[Mark.O].learn(observations[Mark.O], action, reward)
                    players[Mark.X].learn(observations[Mark.X], action, reward)

                players[Mark.O].reset()
                players[Mark.X].reset()
                obs = env.reset()
                break

    td_agent.merge(players[Mark.X])
    td_agent.merge(players[Mark.O])

    return td_agent


def learn(builder: AgentBuilder, num_games: int, num_agents: int, policy_filename: str = None):
    """
    Pit agents against themselves tournament style. The winners survive.
    :param builder: A preset builder with settings necessary for smart agent
    :param num_games:  The number of games to play each other
    :param num_agents:  The number of games to play each other
    :param policy_filename: The filename to save the learned policy to
    """
    processes = multiprocessing.cpu_count()

    if num_agents == 0:
        num_agents = processes

    print(f"Learning! Number of games: {num_games} Number of agents: {num_agents}")

    if num_agents < processes:
        processes = num_agents

    main_agent = builder.make()
    chunksize = math.floor(num_agents / processes)
    with multiprocessing.Pool(processes=processes) as pool:
        agents = [(builder, num_games, i, processes) for i in range(num_agents)]

        print("Playing games...")
        agents = pool.map(learn_from_game, iterable=agents, chunksize=chunksize)

        print("Merging knowledge...")
        for agent in agents:
            main_agent.merge(agent)

    if policy_filename:
        filename = policy_filename
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.getcwd() + "/" + timestamp + ".json"

    save_learning_agent(main_agent, filename=filename)


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="Play a game of TicTacToe or Train the EGreedySampleAveraging agent.")

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    options = parser.parse_args(sys.argv[1:2])
    subparsers = parser.add_subparsers()

    if options.command == "play":
        subparser = subparsers.add_parser("play", help="Play a game of TicTacToe!")
        subparser.add_argument("-X", choices=["human", "base", "smart"], help="Human, Base, or Smart",
                               default="smart")
        subparser.add_argument("-O", choices=["human", "base", "smart"], help="Human, Base, or Smart",
                               default="human")
        subparser.add_argument("-p", "--with-policy", help="A data file containing a policy, generated from learning.")
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])

        agent_types = {"human": AgentBuilder(policy="Human"), "base": AgentBuilder(policy="Random"),
                       "smart": AgentBuilder(policy="EGreedy", learning="TemporalDifferenceAveraging")}

        players = [suboptions.X, suboptions.O]

        for player in players:
            if player == "smart":
                if suboptions.with_policy:
                    state_values, transitions = load_learning_agent(suboptions.with_policy)
                else:
                    state_values, transitions = None, None

                agent_types[player].set(exploratory_rate=0.0,
                                        state_values=state_values,
                                        transitions=transitions)

        play(player_x=agent_types[players[0]].make(), player_o=agent_types[players[1]].make())

    elif options.command == "learn":
        subparser = subparsers.add_parser("learn",
                                          help="Pit two temporal agents against each other and generate a value map.")
        subparser.add_argument("-n", "--num-games",
                               help="The number of games to play against each other.", type=int, default=1000)
        subparser.add_argument("-a", "--num-agents",
                               help="The number of agents to pit against themselves.", type=int, default=0)
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               type=float,
                               default=1.0)
        subparser.add_argument("-l", "--learning-rate",
                               help="The amount to scale the learning of the temporal difference algorithm.",
                               type=float,
                               default=0.15)
        subparser.add_argument("-p", "--with-policy", help="A data file containing a policy, generated from learning.",
                               default=None)
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])

        if suboptions.with_policy:
            state_values, transitions = load_learning_agent(suboptions.with_policy)
        else:
            state_values, transitions = None, None

        builder = AgentBuilder(policy="EGreedy", learning="TemporalDifferenceAveraging")
        builder.set(exploratory_rate=suboptions.exploratory_rate,
                    state_values=state_values,
                    transitions=transitions)
        learn(builder, num_games=suboptions.num_games, num_agents=suboptions.num_agents,
              policy_filename=suboptions.with_policy)
        print("No other options.")


if __name__ == "__main__":
    main()
