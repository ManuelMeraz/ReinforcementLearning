#!/usr/bin/env python3

from __future__ import division, print_function

import argparse
import math
import multiprocessing
import os
import signal
import sys
from collections import defaultdict
from typing import List, Dict

import gym
import matplotlib.pyplot as plt
import numpy
import yaml
from tqdm import tqdm

from rl.agents import AgentBuilder
from rl.agents.reprs import Value
from rl.utils.logging_utils import Logger


def get_builder(agent_config, num_arms, optimistic: float):
    if optimistic > 0:
        state_values = defaultdict(Value)
        for i in range(num_arms):
            state_values[(i,)] = Value(count=0, value=optimistic)
        agent_config["kwargs"]["state_values"] = state_values

    builder = AgentBuilder(agent_config["policy"], agent_config["learning"])
    builder.set(**agent_config["kwargs"])
    return builder


def transition_model(state, action):
    return numpy.array([1]), [numpy.array([action])]


def available_actions(num_arms):
    return numpy.arange(num_arms)


def play(agent, env_name, arms, num_iterations, nonstationary):
    # Create a window to render into
    env = gym.make(env_name)
    obs = env.reset()

    state = numpy.array([env.action_space.sample()])
    rewards = numpy.zeros(num_iterations)
    optimal_percentages = numpy.zeros(num_iterations)
    optimal_count = 0
    for ep in range(num_iterations):
        if nonstationary:
            mu = nonstationary["mu"]
            sigma = nonstationary["sigma"]
            adjustments = numpy.random.normal(mu, sigma, arms)
            for i in range(arms):
                env.r_dist[i][0] += adjustments[i]

        optimal_action = env.r_dist.index(max(env.r_dist))
        action = agent.act(state, available_actions(arms))

        obs, reward, done, info = env.step(action)
        state = numpy.array([action])
        if action == optimal_action:
            optimal_count += 1
        optimal_percentages[ep] = optimal_count / (ep + 1)
        rewards[ep] = reward
        agent.learn(state=state, action=action, reward=reward)

    return rewards, optimal_percentages


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


def plot(data: numpy.ndarray, agent_config, image_name: str, num_iterations: int):
    fig, (rewards_plot, percentage_plot) = plt.subplots(2, 1, figsize=(20, 16))

    iterations = numpy.arange(num_iterations)
    for d, agent in zip(data, agent_config):
        rewards = d[0]
        label = f"{agent['policy']}{agent['learning']}"
        for kwarg, value in agent["kwargs"].items():
            if kwarg != "state_values" and kwarg != "transitions":
                label += f" {kwarg[0]}: {value}"
        rewards_plot.plot(iterations, rewards, label=label)
        optimal_percentages = d[1] * 100
        percentage_plot.plot(iterations, optimal_percentages, label=label)

    rewards_plot.set(xlabel='Iterations', ylabel='Rewards',
                     title='Iterations vs Rewards')

    percentage_plot.set(xlabel='Iterations', ylabel='Optimal %',
                        title='Iterations vs Optimal %')

    percentage_plot.set_ylim(0, 100)

    rewards_plot.legend()
    percentage_plot.legend()
    plt.tight_layout()
    plt.savefig(image_name, dpi=100)
    plt.plot()


def run_experiment(args):
    agent_config = args[0]
    arms = args[1]
    num_episodes = args[2]
    num_iterations = args[3]
    env_name = args[4]
    nonstationary = args[5]

    builder = get_builder(agent_config, arms, agent_config.get("optimistic", 0))

    rewards = numpy.zeros(num_iterations)
    percentages = numpy.zeros(num_iterations)
    label = f"{agent_config['policy']}{agent_config['learning']}"
    for kwarg, value in agent_config["kwargs"].items():
        if kwarg != "state_values" and kwarg != "transitions":
            label += f" {kwarg[0]}: {value}"
    for ep_num in tqdm(range(num_episodes), total=num_episodes, desc=label):
        agent = builder.make()
        agent.transition_model = transition_model
        new_rewards, optimal_percentages = play(agent, env_name, arms, num_iterations=num_iterations,
                                                nonstationary=nonstationary)
        rewards += 1 / (ep_num + 1) * (new_rewards - rewards)
        percentages += 1 / (ep_num + 1) * (optimal_percentages - percentages)

    return rewards, percentages


def simulate(agents: List[Dict], arms: int, env_name: str, num_episodes: int, num_iterations: int, nonstationary: Dict):
    """
    :param num_episodes:  The number of games to play each other
    """
    processes = multiprocessing.cpu_count()
    print(f"Simulating bandits! Number of episodes per agent: {num_episodes} Number of agents: {len(agents)}")
    print(f"Env: {env_name} arms: {arms}")

    if len(agents) < processes:
        processes = len(agents)

    chunksize = math.floor(len(agents) / processes)

    with multiprocessing.Pool(processes=processes) as pool:
        experiment_inputs = ((agent_config, arms, num_episodes, num_iterations, env_name, nonstationary) for
                             agent_config in agents)

        os.system('clear')
        total_rewards_percentages = numpy.array(
            pool.map(run_experiment, iterable=experiment_inputs, chunksize=chunksize))

    return total_rewards_percentages


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("config-file", help="Configuration file to run this program.",
                        type=str)
    parser.add_argument("-ne", "--num-episodes", help="The number of bandit simulations to run",
                        type=int,
                        default=False)
    parser.add_argument("-ni", "--num-iterations", help="The number of iterations per bandit simulation",
                        type=int,
                        default=False)
    parser.add_argument("-o", "--out-image", help="The name of the plot to be output.",
                        type=str,
                        default=False)
    parser.add_argument("-env", "--env-name", help="rlgym environment to load",
                        type=str,
                        default=False)

    logger: Logger = Logger(parser=parser)
    options = parser.parse_args()
    with open(sys.argv[1], 'r') as f:
        if hasattr(yaml, "FullLoader"):
            configuration = yaml.load(f, Loader=yaml.FullLoader)

    if options.num_episodes:
        configuration["num_episodes"] = options.num_episodes

    if options.num_iterations:
        configuration["num_iterations"] = options.num_iterations

    if options.out_image:
        configuration["out_image"] = options.out_image

    if options.env_name:
        configuration["env_name"] = options.env_name

    total_rewards_percentages = simulate(configuration["agents"],
                                         configuration["arms"],
                                         configuration["env_name"],
                                         configuration["num_episodes"],
                                         configuration["num_iterations"],
                                         configuration.get("nonstationary", {}))

    plot(total_rewards_percentages,
         agent_config=configuration["agents"],
         image_name=configuration["out_image"],
         num_iterations=configuration["num_iterations"])

    print("done!")


if __name__ == "__main__":
    main()
