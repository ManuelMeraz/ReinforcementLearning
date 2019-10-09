#!/usr/bin/env python3

from __future__ import division, print_function

from optparse import OptionParser

import gym

from rl.rlgrid import SmartAgent


def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    # Create a window to render into
    obs = env.reset()
    agent = SmartAgent(env.actions)

    while True:
        renderer = env.render()
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        # If the window was closed
        if renderer.window is None:
            break

        if done:
            env.reset()
            env.render()


if __name__ == "__main__":
    main()
