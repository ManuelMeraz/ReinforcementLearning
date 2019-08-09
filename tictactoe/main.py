#!/usr/bin/env python3
import gym

import gym_tictactoe


def play_game(actions, step_fn=input):
    env = gym.make("tictactoe-v0")
    env.reset()

    # Play actions in action profile

    for action in actions:
        print(env.step(action))
        env.render()

        if step_fn:
            step_fn()

    return env


actions = ["1021", "2111", "1221", "2222", "1121"]
_ = play_game(actions, None)
