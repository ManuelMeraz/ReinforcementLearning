#!/usr/bin/env python3
import argparse
import math
import multiprocessing
import os
import signal
import sys
import time
from typing import Dict, Union, List, Tuple

import numpy
from tqdm import tqdm

from rl.envs.tictactoe import TicTacToeEnv, Status, Mark
from rl.tictactoe import HumanAgent, BaseAgent, SmartAgent
from rl.utils.io import load_state_values, save_state_values
from rl.utils.logging import Logger


def play(player_x: Union[HumanAgent, BaseAgent, SmartAgent],
         player_o: Union[HumanAgent, BaseAgent, SmartAgent]):
    """
    Play game of TicTactoe
    :param player_x: Player X
    :param player_o:  Player O
    """
    env = TicTacToeEnv()
    obs: numpy.ndarray = env.reset()

    players: Dict[Mark, Union[HumanAgent, BaseAgent, SmartAgent]] = {
        Mark.X: player_x,
        Mark.O: player_o,
    }

    env.render(mode="human")

    while True:
        current_player: Union[HumanAgent, BaseAgent, SmartAgent] = players[env.current_player]
        action: int = current_player.act(obs)

        obs: numpy.ndarray
        reward: float
        done: bool
        info: Dict[str, Status]
        obs, reward, done, info = env.step(action)
        env.current_player = env.next_player()
        obs[-1] = env.current_player
        env.render(mode="human")

        if done:
            status: Status = info["status"]

            if status == Status.DRAW:
                print("The game was a draw!")
            else:
                winner = {Status.O_WINS: "O", Status.X_WINS: "X"}
                print(f"The winner is {winner[status]}!")

                print("Playing new game.")

            env.render(mode="human")
            obs = env.reset()


def learn_from_game(args):
    td_agent = args[0]
    num_games = args[1]
    index = args[2]
    num_cpus = args[3]

    for _ in tqdm(range(num_games), desc=f"agent: {index}", total=num_games, position=index % num_cpus):
        env = TicTacToeEnv()
        obs: numpy.ndarray = env.reset()

        players: Dict[Mark, SmartAgent] = {
            Mark.X: SmartAgent(td_agent.learning_rate, td_agent.exploratory_rate, td_agent.state_values),
            Mark.O: SmartAgent(td_agent.learning_rate, td_agent.exploratory_rate, td_agent.state_values),
        }

        trajectory: List[Tuple[numpy.ndarray, float]] = [(obs.copy(), 0)]

        while True:
            current_player: Union[HumanAgent, BaseAgent, SmartAgent] = players[env.current_player]
            action: int = current_player.act(obs)

            obs: numpy.ndarray
            reward: float
            done: bool
            info: Dict[str, Status]
            obs, reward, done, info = env.step(action)
            env.current_player = env.next_player()
            current_player.learn(state=obs, reward=reward)
            trajectory.append((obs.copy(), reward))
            obs[-1] = env.current_player


            if done:

                if info["status"] == Status.X_WINS:
                    players[Mark.O].state_values[tuple(trajectory[-2][0])].count -= 1
                    players[Mark.O].previous_state = trajectory[-4][0].copy()
                    players[Mark.O].learn(state=trajectory[-2][0].copy(), reward=-1 * reward)
                elif info["status"] == Status.O_WINS:
                    players[Mark.X].state_values[tuple(trajectory[-2][0])].count -= 1
                    players[Mark.X].previous_state = trajectory[-4][0].copy()
                    players[Mark.X].learn(state=trajectory[-2][0].copy(), reward=-1 * reward)
                else:

                    if env.current_player == Mark.O:
                        players[Mark.X].state_values[tuple(trajectory[-1][0])].count -= 1
                        players[Mark.X].learn(state=trajectory[-1][0].copy(), reward=reward)

                        players[Mark.O].state_values[tuple(trajectory[-2][0])].count -= 1
                        players[Mark.O].learn(state=trajectory[-2][0].copy(), reward=reward)
                    else:
                        players[Mark.X].state_values[tuple(trajectory[-2][0])].count -= 1
                        players[Mark.X].learn(state=trajectory[-2][0].copy(), reward=reward)
                        players[Mark.O].state_values[tuple(trajectory[-1][0])].count -= 1
                        players[Mark.O].learn(state=trajectory[-1][0].copy(), reward=reward)

                td_agent.merge(players[Mark.X])
                td_agent.merge(players[Mark.O])
                break

    return td_agent


def learn(main_agent: SmartAgent, num_games: int, num_agents: int, policy_filename=None):
    """
    Pit agents against themselves tournament style. The winners survive.
    :param main_agent: The agent that will learn from playing games
    :param num_games:  The number of games to play each other
    :param num_agents:  The number of games to play each other
    """
    processes = multiprocessing.cpu_count()

    if num_agents == 0:
        num_agents = processes

    print(f"Learning! Number of games: {num_games} Number of agents: {num_agents}")

    if num_agents < processes:
        processes = num_agents

    chunksize = math.floor(num_agents / processes)
    with multiprocessing.Pool(processes=processes) as pool:
        agents = [
            (SmartAgent(main_agent.learning_rate, main_agent.exploratory_rate, main_agent.state_values),
             num_games, i, processes) for i in range(num_agents)]

        print("Playing games...")
        agents = pool.map(learn_from_game, iterable=agents, chunksize=chunksize)

        print("Merging knowledge...")
        for agent in agents:
            main_agent.merge(agent)

    if policy_filename:
        filename = policy_filename
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.getcwd() + "/" + timestamp + ".policy"

    save_state_values(main_agent.state_values, filename=filename)


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    parser = argparse.ArgumentParser();
    parser.add_argument("command", help="Play a game of TicTacToe or Train the SmartAgent agent.")

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

        agent_types = {"human": HumanAgent(), "base": BaseAgent(),
                       "smart": SmartAgent(learning_rate=0.5, exploratory_rate=0.0)}

        player_x = agent_types[suboptions.X]
        player_o = agent_types[suboptions.O]

        if isinstance(player_o, SmartAgent) and suboptions.with_policy:
            player_o.state_values = load_state_values(suboptions.with_policy)
            player_o.mark = 'O'

        if isinstance(player_x, SmartAgent) and suboptions.with_policy:
            player_x.state_values = load_state_values(suboptions.with_policy)
            player_o.mark = 'X'

        play(player_x=player_x, player_o=player_o)

    elif options.command == "learn":
        subparser = subparsers.add_parser("learn",
                                          help="Pit two temporal agents against each other and generate a value map.")
        subparser.add_argument("-n", "--num-games",
                               help="The number of games to play against each other.", type=int, default=1000)
        subparser.add_argument("-a", "--num-agents",
                               help="The number of agents to pit against themselves.", type=int, default=0)
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               type=float,
                               default=0.1)
        subparser.add_argument("-l", "--learning-rate",
                               help="The amount to scale the learning of the temporal difference algorithm.",
                               type=float,
                               default=0.5)
        subparser.add_argument("-p", "--with-policy", help="A data file containing a policy, generated from learning.",
                               default=None)
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])

        if suboptions.with_policy:
            policy = load_state_values(suboptions.with_policy)
        else:
            policy = None
        agent = SmartAgent(learning_rate=suboptions.learning_rate, exploratory_rate=suboptions.exploratory_rate,
                           state_values=policy)

        learn(agent, num_games=suboptions.num_games,
              num_agents=suboptions.num_agents, policy_filename=suboptions.with_policy)
        print("No other options.")


if __name__ == "__main__":
    main()
