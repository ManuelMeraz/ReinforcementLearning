#!/usr/bin/env python3

from __future__ import division, print_function

import argparse
import math
import multiprocessing
import os
import signal
import sys
import time

import gym
import numpy
from tqdm import tqdm

from rl.agents import AgentBuilder
from rl.utils.io_utils import save_learning_agent, load_learning_agent
from rl.utils.logging_utils import Logger


def available_actions():
    return numpy.arange(7)


def get_state(obs, env):
    state = numpy.concatenate((obs["image"], obs["direction"],), axis=None)
    # if env.carrying is None:
    #     carrying = 0
    # else:
    #     carrying = 1
    # state = numpy.concatenate((env.agent_pos, env.agent_dir, carrying), axis=None)
    # state = numpy.concatenate((obs["image"], env.agent_pos, obs["direction"],), axis=None)
    return state


def learn_from_game(args):
    builder = args[0]
    num_games = args[1]
    env_name = args[2]
    agent_id = args[3]

    env = gym.make(env_name)

    agent = builder.make()
    obs: numpy.ndarray = env.reset()
    state = get_state(obs, env)
    for _ in tqdm(range(num_games), desc=f"agent: {agent_id}", total=num_games):
        while True:
            action: int = agent.act(state, available_actions=available_actions())

            obs, reward, done, info = env.step(action)

            prior_state = state
            state = get_state(obs, env)
            agent.learn(state=prior_state, action=action, reward=reward)

            if done:
                agent.learn(state=state, action=action, reward=reward)
                agent.reset()
                obs = env.reset()
                state = get_state(obs, env)
                break

    return agent


def learn(builder: AgentBuilder, env_name: str, num_episodes: int, num_agents: int, policy_filename: str):
    """
    Pit agents against themselves tournament style. The winners survive.
    :param num_episodes:  The number of games to play each other
    :param num_agents:  The number of games to play each other
    :param policy_filename: The filename to save the learned policy to
    """
    processes = multiprocessing.cpu_count()

    if num_agents == 0:
        num_agents = processes

    print(f"Learning! Number of games: {num_episodes} Number of agents: {num_agents}")

    if num_agents < processes:
        processes = num_agents

    main_agent = builder.make()
    chunksize = math.floor(num_agents / processes)

    with multiprocessing.Pool(processes=processes) as pool:
        agents = ((builder, num_episodes, env_name, agent_id + 1, num_agents) for agent_id in range(num_agents))

        os.system('clear')
        print("Playing games...")
        agents = pool.map(learn_from_game, iterable=agents, chunksize=chunksize)

        print("Merging knowledge...")
        for agent in tqdm(agents, desc="Merging agents", total=num_agents):
            main_agent.merge(agent)

    policy_filename = os.path.join("policies", f"{env_name}.pickle")
    if os.path.exists("./policies"):
        save_learning_agent(agent, policy_filename)
    else:
        save_learning_agent(agent, f"{env_name}.pickle")


def play(agent, env, episodes=100):
    # Create a window to render into
    obs = env.reset()
    state = get_state(obs, env)

    for _ in range(episodes):
        while True:
            renderer = env.render()
            action: int = agent.act(state, available_actions=available_actions())
            prior_state = state
            obs, reward, done, info = env.step(action)
            state = get_state(obs, env)
            agent.learn(state=prior_state, action=action, reward=reward)

            # # If the window was closed
            if renderer.window is None:
                break
            time.sleep(0.1)

            if done:
                agent.learn(state=state, action=action, reward=reward)
                agent.reset()
                obs = env.reset()
                state = get_state(obs, env)
                break


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
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               type=float,
                               default=0.05)
        subparser.add_argument("-l", "--learning-rate", help="The learning rate for TD learning.",
                               type=float,
                               default=0.8)
        subparser.add_argument("-d", "--discount-rate",
                               help="How much of the current value to discount from the previous value.",
                               type=float,
                               default=0.5)
        subparser.add_argument("-p", "--with-policy", help="A data file containing a policy, generated from learning.")
        subparser.add_argument(
            "-env",
            "--env-name",
            dest="env_name",
            help="rlgym environment to load",
            default='MiniGrid-Empty-5x5-v0'
        )

        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])
        env = gym.make(suboptions.env_name)

        builder = AgentBuilder(policy="EGreedy", learning="TemporalDifferenceZero")
        policy_filename = os.path.join("policies", f"{suboptions.env_name}.pickle")
        if suboptions.with_policy:
            state_values, transitions = load_learning_agent(suboptions.with_policy)
        elif os.path.exists(policy_filename):
            print(f"Using policy: {policy_filename}")
            state_values, transitions = load_learning_agent(policy_filename)
        else:
            state_values, transitions = None, None

        builder.set(exploratory_rate=suboptions.exploratory_rate,
                    learning_rate=suboptions.learning_rate,
                    discount_rate=suboptions.discount_rate,
                    state_values=state_values,
                    transitions=transitions)

        agent = builder.make()
        play(agent, env)

    elif options.command == "learn":
        subparser = subparsers.add_parser("learn",
                                          help="Pit two temporal agents against each other and generate a value map.")
        subparser.add_argument("-ne", "--num-episodes",
                               help="The number of episodes to play against each other.", type=int, default=100)
        subparser.add_argument("-na", "--num-agents",
                               help="The number of agents to pit against themselves.", type=int, default=0)
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               type=float,
                               default=0.1)
        subparser.add_argument("-l", "--learning-rate", help="The learning rate for TD learning.",
                               type=float,
                               default=0.5)
        subparser.add_argument("-d", "--discount-rate",
                               help="How much of the current value to discount from the previous value.",
                               type=float,
                               default=0.5)
        subparser.add_argument("-p", "--with-policy", help="A data file containing a policy, generated from learning.")
        subparser.add_argument(
            "-env",
            "--env-name",
            dest="env_name",
            help="rlgym environment to load",
            default='MiniGrid-Empty-5x5-v0'
        )
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])
        env = gym.make(suboptions.env_name)
        builder = AgentBuilder(policy="EGreedy", learning="TemporalDifferenceZero")
        policy_filename = os.path.join("policies", f"{suboptions.env_name}.pickle")
        if suboptions.with_policy:
            state_values, transitions = load_learning_agent(suboptions.with_policy)
        elif os.path.exists(policy_filename):
            print(f"Using policy: {policy_filename}")
            state_values, transitions = load_learning_agent(policy_filename)
        else:
            state_values, transitions = None, None

        builder.set(exploratory_rate=suboptions.exploratory_rate,
                    learning_rate=suboptions.learning_rate,
                    discount_rate=suboptions.discount_rate,
                    state_values=state_values,
                    transitions=transitions)

        learn(builder=builder,
              env_name=suboptions.env_name,
              num_episodes=suboptions.num_episodes,
              num_agents=suboptions.num_agents,
              policy_filename=policy_filename)


if __name__ == "__main__":
    main()
