#!/usr/bin/env python3

from __future__ import division, print_function

import argparse
import math
import signal
import sys

import gym
import matplotlib.pyplot as plt
import numpy
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from rl.bandits import SmartAgent
from rl.reprs import Transition
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

    agent = SmartAgent(agent.action_space, agent.exploratory_rate)
    env.reset()
    return rewards


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-episodes", help="The number of times to sample from bandits",
                        type=int,
                        default=3000)
    parser.add_argument("-e", "--exploratory-rate", help="The probability of exploring rather than exploiting.",
                        type=float,
                        default=0.1)
    parser.add_argument(
        "-env",
        "--env-name",
        dest="env_name",
        help="rlgym environment to load",
        default='BanditTenArmedGaussian-v0'
    )

    logger: Logger = Logger(parser=parser)
    options = parser.parse_args()

    env = gym.make(options.env_name)
    agents = [SmartAgent(action_space=env.action_space, exploratory_rate=0.0),
              SmartAgent(action_space=env.action_space, exploratory_rate=0.01),
              SmartAgent(action_space=env.action_space, exploratory_rate=0.10)]

    fig, ax = plt.subplots(figsize=(10, 8))

    num_iterations = 2000
    for agent in agents:
        rewards = numpy.zeros(num_iterations)
        for run in range(options.num_episodes):
            new_rewards = play(agent, options.env_name, num_episodes=num_iterations)
            for i in range(num_iterations):
                rewards[i] += (1 / (run + 1)) * (new_rewards[i] - rewards[i])

        iterations = numpy.arange(num_iterations)
        ax.plot(iterations, rewards, label=str(agent.exploratory_rate))

    ax.set(xlabel='iterations', ylabel='rewards',
           title='iterations vs rewards')

    ax.set_xlim(0, num_iterations)
    ax.set_ylim(-0.2, 1.5)

    ax.xaxis.set_major_locator(MultipleLocator(math.floor(0.2 * num_iterations)))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    ax.xaxis.set_minor_locator(AutoMinorLocator(math.floor(0.04 * num_iterations)))
    ax.yaxis.set_minor_locator(AutoMinorLocator(0.02))
    ax.grid()
    ax.legend()
    plt.savefig('10armedtestbed.png', dpi=100)
    print("done!")

if __name__ == "__main__":
    main()
