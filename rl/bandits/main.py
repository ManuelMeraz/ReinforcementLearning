#!/usr/bin/env python3

from __future__ import division, print_function

import argparse
import signal
import sys
from collections import defaultdict, Counter

import gym
import matplotlib.pyplot as plt
import numpy

from rl.bandits import EGreedySampleAveraging, EGreedyWeightedAveraging
from rl.reprs import Transition, Value
from rl.utils.logging import Logger


def play(agent, env_name, num_episodes=1000):
    # Create a window to render into
    env = gym.make(env_name)
    obs = env.reset()
    state = numpy.array([env.action_space.sample()])
    rewards = numpy.zeros(num_episodes)
    for ep in range(num_episodes):
        action = agent.act(state)

        obs, reward, done, info = env.step(action)
        state = numpy.array([action])
        rewards[ep] = reward
        transition = Transition(state=state, action=action, reward=reward)
        agent.learn(transition)
        # print(f"action: {action} reward: {reward} episodes: {ep + 1}")

    agent.reset()
    agent.state_values = defaultdict(Value)
    agent.transitions = defaultdict(Counter)
    return rewards


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


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

    parser.add_argument("-o", "--out-image", help="The name of the plot to be output.",
                        type=str,
                        default="ArmedBandits")
    parser.add_argument(
        "-env",
        "--env-name",
        dest="env_name",
        help="rlgym environment to load",
        default='BanditTenArmedGaussian-v0'
    )

    logger: Logger = Logger(parser=parser)
    options = parser.parse_args()

    agent_types = {
        "EGSA": EGreedySampleAveraging,
        "EGWA": EGreedyWeightedAveraging
    }
    env = gym.make(options.env_name)
    constructor_kwargs = {
        "action_space": env.action_space,
    }

    if options.agent == "EGWA" or options.agent == "DEGWA":
        constructor_kwargs["learning_rate"] = options.learning_rate

    exploratory_rates = [0.0, 0.01, 0.1]

    agents = []
    for e in exploratory_rates:
        constructor_kwargs["exploratory_rate"] = e
        agents.append(agent_types[options.agent](**constructor_kwargs))

    fig, ax = plt.subplots(figsize=(10, 8))

    for agent in agents:
        rewards = numpy.zeros(options.num_iterations)
        for run in range(options.num_episodes):
            new_rewards = play(agent, options.env_name, num_episodes=options.num_iterations)
            for i in range(options.num_iterations):
                rewards[i] += (1 / (run + 1)) * (new_rewards[i] - rewards[i])

        iterations = numpy.arange(options.num_iterations)
        ax.plot(iterations, rewards, label=str(agent.exploratory_rate))

    ax.set(xlabel='iterations', ylabel='rewards',
           title='iterations vs rewards')

    ax.set_xlim(0, options.num_iterations)
    ax.set_ylim(-0.2, 1.5)

    ax.legend()
    plt.savefig(options.out_image, dpi=100)
    print("done!")


if __name__ == "__main__":
    main()
