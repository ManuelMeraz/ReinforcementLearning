#!/usr/bin/env python3

from __future__ import division, print_function

import argparse
import math
import multiprocessing
import os
import signal
import sys
from collections import defaultdict
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm

from rl.agents import AgentBuilder
from rl.agents.reprs import Value
from rl.utils.logging_utils import Logger


def transition_model(state, action):
    return numpy.array([1]), [numpy.array([action])]


def available_actions():
    return numpy.arange(10)


def play(agent, env_name, num_iterations, nonstationary):
    # Create a window to render into
    env = gym.make(env_name)
    obs = env.reset()

    state = numpy.array([env.action_space.sample()])
    rewards = numpy.zeros(num_iterations)
    optimal_percentages = numpy.zeros(num_iterations)
    optimal_count = 0
    for ep in range(num_iterations):
        if nonstationary:
            adjustments = numpy.random.normal(0, 0.1, 10)
            for i in range(10):
                env.r_dist[i][0] += adjustments[i]

        optimal_action = env.r_dist.index(max(env.r_dist))
        action = agent.act(state, available_actions())

        obs, reward, done, info = env.step(action)
        state = numpy.array([action])
        if action == optimal_action:
            optimal_count += 1
        optimal_percentages[ep] = optimal_count / (ep + 1)
        rewards[ep] = reward
        agent.learn(state=state, action=action, reward=reward)
    # print(f"action: {action} reward: {reward}")
    # time.sleep(0.1)
    return rewards, optimal_percentages


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


def plot(data: numpy.ndarray, exploratory_rates, image_name: str, num_iterations: int):
    fig, (rewards_plot, percentage_plot) = plt.subplots(2, 1, figsize=(10, 8))

    iterations = numpy.arange(num_iterations)
    for d, e in zip(data, exploratory_rates):
        rewards = d[0]
        rewards_plot.plot(iterations, rewards, label=str(e))
        optimal_percentages = d[1] * 100
        percentage_plot.plot(iterations, optimal_percentages, label=str(e))

    rewards_plot.set(xlabel='Iterations', ylabel='Rewards',
                     title='Iterations vs Rewards')

    percentage_plot.set(xlabel='Iterations', ylabel='Optimal %',
                        title='Iterations vs Optimal %')

    percentage_plot.set_ylim(0, 100)

    rewards_plot.legend()
    percentage_plot.legend()
    plt.tight_layout()
    plt.savefig(image_name, dpi=100)


def run_experiment(args):
    builder = args[0]
    num_episodes = args[1]
    num_iterations = args[2]
    env_name = args[3]
    nonstationary = args[4]

    agent = builder.make()
    rewards = numpy.zeros(num_iterations)
    percentages = numpy.zeros(num_iterations)
    for ep_num in tqdm(range(num_episodes), total=num_episodes, desc=f"agent: {agent.exploratory_rate}"):
        agent.transition_model = transition_model
        new_rewards, optimal_percentages = play(agent, env_name, num_iterations=num_iterations,
                                                nonstationary=nonstationary)
        rewards += 1 / (ep_num + 1) * (new_rewards - rewards)
        percentages += 1 / (ep_num + 1) * (optimal_percentages - percentages)
        agent = builder.make()

    return rewards, percentages


def simulate(builders: List[AgentBuilder], env_name: str, num_episodes: int, num_iterations: int, nonstationary: bool):
    """
    :param num_episodes:  The number of games to play each other
    """
    processes = multiprocessing.cpu_count()

    print(f"Simulating bandits! Number of episodes per agent: {num_episodes} Number of agents: {len(builders)}")

    if len(builders) < processes:
        processes = len(builders)

    chunksize = math.floor(len(builders) / processes)

    with multiprocessing.Pool(processes=processes) as pool:
        experiment_inputs = ((builder, num_episodes, num_iterations, env_name, nonstationary) for builder in builders)

        os.system('clear')
        print("Playing games...")
        total_rewards_percentages = numpy.array(
            pool.map(run_experiment, iterable=experiment_inputs, chunksize=chunksize))

    return total_rewards_percentages


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("-ne", "--num-episodes", help="The number of bandit simulations to run",
                        type=int,
                        default=2000)
    parser.add_argument("-ni", "--num-iterations", help="The number of iterations per bandit simulation",
                        type=int,
                        default=1000)
    parser.add_argument("-a", "--agent", help="The type of agent to use", choices=["EGSA", "EGWA", "DEGSA", "DEGWA"],
                        type=str,
                        default="EGSA")
    parser.add_argument("-l", "--learning-rate", help="The learning rate for weighted averaging learning.",
                        type=float,
                        default=0.1)
    parser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                        type=float,
                        default=0.1)
    parser.add_argument("-ns", "--nonstationary", help="Stationary bandits.",
                        action="store_true",
                        default=False)
    parser.add_argument("-op", "--optimistic", help="Make the exploratory rate=0.0 agent start with optimistic values",
                        action="store_true",
                        default=False)
    parser.add_argument("-o", "--out-image", help="The name of the plot to be output.",
                        type=str,
                        default="ArmedBandits")
    parser.add_argument("-env", "--env-name", help="rlgym environment to load",
                        type=str,
                        default='BanditTenArmedGaussian-v0')

    logger: Logger = Logger(parser=parser)
    options = parser.parse_args()

    agent_types = {
        "EGSA": ("EGreedy", "SampleAveraging"),
        "EGWA": ("EGreedy", "WeightedAveraging"),
    }

    constructor_kwargs = {}
    if options.agent == "EGWA" or options.agent == "DEGWA":
        constructor_kwargs["learning_rate"] = options.learning_rate

    exploratory_rates = [0.0, 0.1]

    builders = []
    for e in exploratory_rates:
        if options.optimistic and e == 0.0:
            state_values = defaultdict(Value)
            for i in range(10):
                state_values[(i,)] = Value(count=0, value=5)
            constructor_kwargs["state_values"] = state_values

        constructor_kwargs["exploratory_rate"] = e
        builder = AgentBuilder(*agent_types[options.agent])
        builder.set(**constructor_kwargs)
        if options.optimistic and e == 0.0:
            del constructor_kwargs["state_values"]
        builders.append(builder)

    total_rewards_percentages = simulate(builders, options.env_name, options.num_episodes, options.num_iterations,
                                         options.nonstationary)
    plot(total_rewards_percentages, exploratory_rates=exploratory_rates, image_name=options.out_image,
         num_iterations=options.num_iterations)

    print("done!")


if __name__ == "__main__":
    main()
