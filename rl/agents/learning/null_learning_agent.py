#! /usr/bin/env python3

from rl.agents.learning import LearningAgent


class NullLearningAgent(LearningAgent):
    """
    The learning agent implements a learning method and is used for purposes of building a state value map
    """

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    def learn_value(self):
        pass
