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


def play(player_x: Union[Human, Base, TemporalDifference], player_o: Union[Human, Base, TemporalDifference]):
    """
    Play game of TicTactoe
    :param player_x: Player X
    :param player_o:  Player O
    """
    env = TicTacToeEnv()
    obs: Tuple[Mark] = env.reset()

    players: Dict[str, Union[Human, Base, TemporalDifference]] = {
        "X": player_x,
        "O": player_o,
    }

    mode = 'human' if isinstance(player_x, Human) or isinstance(player_o, Human) else None

    while True:
        env.render(mode=mode)
        current_player: Union[Human, Base, TemporalDifference] = players[obs["next_player"]]
        action: int = current_player.act(obs["board"])

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


def learn_from_game(args):
    td_agent = args[0]
    num_games = args[1]
    index = args[2]
    num_cpus = args[3]

    for _ in tqdm(range(num_games), desc=f"agent: {index}", total=num_games, position=index % num_cpus):
        env = TicTacToeEnv()
        obs: Tuple[Mark] = env.reset()

        players: Dict[str, TemporalDifference] = {
            "X": TemporalDifference('X', td_agent.exploratory_rate, td_agent.learning_rate, td_agent.state_values),
            "O": TemporalDifference('O', td_agent.exploratory_rate, td_agent.learning_rate, td_agent.state_values),
        }

        while True:
            current_player = obs["next_player"]
            action: int = players[current_player].act(obs["board"])

            prev_obs = obs
            obs, reward, done, info = env.step(action)
            players[current_player].learn(board_state=obs["board"], reward=reward)

            if done:

                if info["status"] == Status.X_WINS:
                    players["O"].learn(board_state=prev_obs["board"], reward=-1 * reward)
                elif info["status"] == Status.O_WINS:
                    players["X"].learn(board_state=prev_obs["board"], reward=-1 * reward)
                else:
                    players["X"].learn(board_state=prev_obs["board"], reward=reward)
                    players["O"].learn(board_state=prev_obs["board"], reward=reward)


                td_agent.merge(players["X"])
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
        agents = [
            (TemporalDifference('X', main_agent.exploratory_rate, main_agent.learning_rate, main_agent.state_values),
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
        subparser.add_argument("-p", "--with-policy", help="A data file containing a policy, generated from learning.")
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])

        agentTypes = {"human": Human(), "base": Base(),
                      "td": TemporalDifference('X', exploratory_rate=0.0, learning_rate=0.5)}

        player_x = agentTypes[suboptions.X]
        player_o = agentTypes[suboptions.O]

        if isinstance(player_o, TemporalDifference) and suboptions.with_policy:
            player_o.state_values = TemporalDifference.load_state_values(suboptions.with_policy)
            player_o.mark = 'O'

        if isinstance(player_x, TemporalDifference) and suboptions.with_policy:
            player_x.state_values = TemporalDifference.load_state_values(suboptions.with_policy)
            player_o.mark = 'X'

        play(player_x=player_x, player_o=player_o)


    elif options.command == "learn":
        subparser = subparsers.add_parser("learn",
                                          help="Pit two temporal agents against each other and generate a value map.")
        subparser.add_argument("-n", "--num-games",
                               help="The number of games to play against each other.", type=int, default=10000)
        subparser.add_argument("-a", "--num-agents",
                               help="The number of agents to pit against themselves.", type=int, default=24)
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               type=float,
                               default=0.1)
        subparser.add_argument("-l", "--learning-rate",
                               help="The amount to scale the learning of the temporal difference algorithm.",
                               type=float,
                               default=0.5)
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])

        agent = TemporalDifference('X', exploratory_rate=suboptions.exploratory_rate,
                                   learning_rate=suboptions.learning_rate)
        learn(agent, num_games=suboptions.num_games, num_agents=suboptions.num_agents)
    else:
        print("No other options.")


if __name__ == "__main__":
    main()
