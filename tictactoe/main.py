#!/usr/bin/env python3
from gym_tictactoe.env import (O_REWARD, X_REWARD, TicTacToeEnv,
                               after_action_state, agent_by_mark,
                               check_game_status, next_mark, set_log_level_by)
from collections import defaultdict
from temporal_difference import TemporalDifference
import pandas as pd
import sys


DEFAULT_VALUE = 0
EPISODE_CNT = 17000
BENCH_EPISODE_CNT = 3000
MODEL_FILE = "best_td_agent.dat"
LEARNING_RATE = 0.08
ALPHA = 0.4

st_values = {}
st_visits = defaultdict(lambda: 0)

def play(td_gent, vs_agent, show_number):
    env = TicTacToeEnv(show_number=show_number)
    td_agent = TemporalDifference("X", 0, 0)  # prevent exploring
    start_mark = "O"
    agents = [vs_agent, td_agent]

    while True:
        # start agent rotation
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False

        # show start board for human agent

        if mark == "O":
            env.render(mode="human")

        while not done:
            agent = agent_by_mark(agents, mark)
            human = isinstance(agent, HumanAgent)

            env.show_turn(True, mark)
            ava_actions = env.available_actions()

            if human:
                action = agent.act(ava_actions)

                if action is None:
                    sys.exit()
            else:
                action = agent.act(state, ava_actions)

            state, reward, done, info = env.step(action)

            env.render(mode="human")

            if done:
                env.show_result(True, mark, reward)

                break
            else:
                _, mark = state

        # rotation start
        start_mark = next_mark(start_mark)


def learn(max_episode, epsilon, learning_rate, save_file):
    reset_state_values()

    env = TicTacToeEnv()
    agents = [TemporalDifference("O", epsilon, learning_rate), TemporalDifference("X", epsilon, learning_rate)]

    start_mark = "O"

    for i in range(max_episode):
        episode = i + 1
        env.show_episode(False, episode)

        # reset agent for new episode

        for agent in agents:
            agent.episode_rate = episode / float(max_episode)

        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False

        while not done:
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            env.show_turn(False, mark)
            action = agent.act(state, ava_actions)

            # update (no rendering)
            nstate, reward, done, info = env.step(action)
            agent.backup(state, nstate, reward)

            if done:
                env.show_result(False, mark, reward)
                # set terminal state value
                set_state_value(state, reward)

            _, mark = state = nstate

        # rotate start
        start_mark = next_mark(start_mark)

    # save states
    save_model(save_file, max_episode, epsilon, learning_rate)

if __name__ == "__main__":
    

