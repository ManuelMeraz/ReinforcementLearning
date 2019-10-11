#!/usr/bin/env python3

from __future__ import division, print_function

import argparse
import signal
import sys
import time

import gym
import numpy

from rl.bandits import SmartAgent
from rl.reprs import Transition
from rl.utils.logging import Logger


def play(agent, env, num_episodes=1000):
    # Create a window to render into
    obs = env.reset()
    state = numpy.array([env.action_space.sample()])

    for ep in range(num_episodes):
        env.render()
        action = agent.act(state)

        obs, reward, done, info = env.step(action)
        state = numpy.array([action])
        transition = Transition(state=state, action=action, reward=reward)
        agent.learn(transition)
        print(f"action: {action} reward: {reward} episodes: {ep + 1}")

        time.sleep(0.01)


def keyboard_interrupt_handler(signal, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-episodes", help="The number of times to sample from bandits",
                        type=int,
                        default=10000)
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
    agent = SmartAgent(action_space=env.action_space, exploratory_rate=options.exploratory_rate)
    play(agent, env, num_episodes=options.num_episodes)


if __name__ == "__main__":
    main()
