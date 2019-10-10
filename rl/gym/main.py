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

from rl.reprs import Transition
from rl.gym import SmartAgent, HumanAgent, BaseAgent
from rl.utils.io import save_learning_agent, load_learning_agent
from rl.utils.logging import Logger


def get_state(obs, env):
    return obs


def learn_from_game(args):
    agent = args[0]
    num_games = args[1]
    index = args[2]
    num_cpus = args[3]
    env_name = args[4]

    env = gym.make(env_name)

    obs: numpy.ndarray = env.reset()
    state = get_state(obs, env)

    for _ in tqdm(range(num_games), desc=f"agent: {index}", total=num_games, position=index % num_cpus):
        while True:
            action: int = agent.act(state)

            obs, reward, done, info = env.step(action)

            prior_state = state
            state = get_state(obs, env)
            transition = Transition(state=prior_state, action=action, reward=reward)
            agent.learn(transition)

            if done:
                agent.reset()
                obs = env.reset()
                state = get_state(obs, env)
                break

    return agent


def learn(main_agent: SmartAgent, env_name: str, num_episodes: int, num_agents: int, policy_filename: str):
    """
    Pit agents against themselves tournament style. The winners survive.
    :param main_agent: The agent that will learn from playing games
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

    chunksize = math.floor(num_agents / processes)
    with multiprocessing.Pool(processes=processes) as pool:
        agents = [
            (SmartAgent(action_space=main_agent.actions, exploratory_rate=main_agent.exploratory_rate,
                        learning_rate=main_agent.learning_rate, state_values=main_agent.state_values,
                        transitions=main_agent.transitions),
             num_episodes, i, processes, env_name) for i in range(num_agents)]

        print("Playing games...")
        agents = pool.map(learn_from_game, iterable=agents, chunksize=chunksize)

        print("Merging knowledge...")
        for agent in agents:
            main_agent.merge(agent)

    policy_filename = os.path.join("policies", f"{env_name}.json")
    if os.path.exists("./policies"):
        save_learning_agent(agent, policy_filename)
    else:
        save_learning_agent(agent, f"{env_name}.json")


def play(agent, env, episodes=100):
    # Create a window to render into
    obs = env.reset()
    state = get_state(obs, env)

    for _ in range(episodes):
        while True:
            env.render()
            action = agent.act(state)
            prior_state = state
            obs, reward, done, info = env.step(action)
            state = get_state(obs, env)
            transition = Transition(state=prior_state, action=action, reward=reward)
            agent.learn(transition)

            time.sleep(0.1)

            if done:
                agent.reset()
                obs = env.reset()
                state = get_state(obs, env)
                break


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
        subparser.add_argument("-a", "--agent", choices=["human", "base", "smart"], help="Human, Base, or Smart",
                               default="smart")
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               type=float,
                               default=0.05)
        subparser.add_argument("-l", "--learning-rate", help="The learning rate for TD learning.",
                               type=float,
                               default=0.8)
        subparser.add_argument("-p", "--with-policy", help="A data file containing a policy, generated from learning.")
        subparser.add_argument(
            "-env",
            "--env-name",
            dest="env_name",
            help="gym environment to load",
            default='CartPole-v0'
        )

        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])
        env = gym.make(suboptions.env_name)

        agent_types = {"human": HumanAgent(), "base": BaseAgent(env.action_space),
                       "smart": SmartAgent(action_space=env.action_space, exploratory_rate=suboptions.exploratory_rate,
                                           learning_rate=suboptions.learning_rate)}

        agent = agent_types[suboptions.agent]

        policy_filename = os.path.join("policies", f"{suboptions.env_name}.json")
        if suboptions.with_policy:
            agent.state_values, agent.transitions = load_learning_agent(suboptions.with_policy)
        elif isinstance(agent, SmartAgent) and os.path.exists(policy_filename):
            print(f"Using policy: {policy_filename}")
            agent.state_values, agent.transitions = load_learning_agent(policy_filename)

        play(agent, env)

    elif options.command == "learn":
        subparser = subparsers.add_parser("learn",
                                          help="Pit two temporal agents against each other and generate a value map.")
        subparser.add_argument("-ne", "--num-episodes",
                               help="The number of episodes to play against each other.", type=int, default=100)
        subparser.add_argument("-na", "--num-agents",
                               help="The number of agents to pit against themselves.", type=int, default=0)
        subparser.add_argument("-a", "--agent", choices=["human", "base", "smart"], help="Human, Base, or Smart",
                               default="smart")
        subparser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                               type=float,
                               default=0.5)
        subparser.add_argument("-l", "--learning-rate", help="The learning rate for TD learning.",
                               type=float,
                               default=0.5)
        subparser.add_argument("-p", "--with-policy", help="A data file containing a policy, generated from learning.")
        subparser.add_argument(
            "-env",
            "--env-name",
            dest="env_name",
            help="gym environment to load",
            default='CartPole-v0'
        )
        logger: Logger = Logger(parser=subparser)

        suboptions = subparser.parse_args(sys.argv[2:])
        env = gym.make(suboptions.env_name)

        if suboptions.with_policy:
            policy_filename = suboptions.with_policy
        elif os.path.exists("./policies"):
            policy_filename = os.path.join("policies", f"{suboptions.env_name}.json")
        else:
            policy_filename = f"{suboptions.env_name}.json"

        if os.path.exists(os.path.expanduser(policy_filename)):
            print(f"Using policy: {policy_filename}")
            state_values, transitions = load_learning_agent(policy_filename)
        else:
            state_values, transitions = None, None

        agent = SmartAgent(action_space=env.action_space, exploratory_rate=suboptions.exploratory_rate,
                           learning_rate=suboptions.learning_rate,
                           state_values=state_values, transitions=transitions)

        learn(agent, suboptions.env_name, num_episodes=suboptions.num_episodes, num_agents=suboptions.num_agents,
              policy_filename=policy_filename)


if __name__ == "__main__":
    main()
