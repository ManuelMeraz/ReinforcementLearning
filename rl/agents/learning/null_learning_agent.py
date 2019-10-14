#! /usr/bin/env python3

from rl.agents.learning import LearningAgent


class NullLearning(LearningAgent):
    """
    The null learning agent is an agent that does nothing. This is used for the agent builder when a learning
    agent is not required, and only a policy agent is desired for building an agent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn_value(self):
        pass
