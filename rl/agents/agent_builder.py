#! /usr/bin/env python3

import logging
from typing import List, Dict

from rl import agents
from rl.agents import Agent


class AgentBuilder:
    """
    The agent builder class is used to dynamically build and create new agent classes from merging a learning agent
    with some policy agent.
    """
    is_set: bool
    learning_agent: str
    learning_agent_names: List[str]
    policy_agent: str
    policy_agent_names: List[str]
    registry: Dict[str, type]

    def __init__(self, policy: str = "NullPolicy", learning: str = "NullLearning"):
        """
        The builder may either be constructed with the strings pertaining to it's corresponding policy or learning
        agent class names or they may be added after it's constructed.
        :param policy: The name a policy agent
        :param learning: The name of a learning agent
        """
        learning_agents: List[type] = agents.LearningAgent.__subclasses__()
        policy_agents: List[type] = agents.PolicyAgent.__subclasses__()

        self.learning_agent_names = [agent.__name__ for agent in learning_agents]
        self.policy_agent_names = [agent.__name__ for agent in policy_agents]

        agent_subclasses: List[type] = learning_agents + policy_agents
        agent_subclass_names: List[str] = [agent.__name__ for agent in agent_subclasses]
        self.registry = dict(zip(agent_subclass_names, agent_subclasses))

        self.learning_agent = learning
        self.policy_agent = policy
        self.is_set = False

        # Constructors arguments
        self.args = []
        self.kwargs = {}

    def add(self, agent_type: str):
        """
        Used to add the agent classes used to build the new agent. Can either be a learning agent or policy agent.
        builder.add("TemporalDifferenceZero") is a learning agent
        builder.add("Random") is a policy agent
        Will produce a RandomTemporalDifferenceZero agent

        :param agent_type: The type of agent that will be built
        """
        assert agent_type in self.registry, "Invalid agent type"
        assert self.learning_agent == "NullLearning" or self.policy_agent == "NullPolicy", \
            "Cannot add in more agent types."

        if agent_type in self.learning_agent_names:
            self.learning_agent = agent_type
        else:
            self.policy_agent = agent_type

    def set(self, *args, **kwargs):
        """
        These are the constructor arguments required to build both classes that are being merged. Using keyword arguments
        is the way to go. Agents that do not required constructor argument may be made without setting the arguments.
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        """
        self.args = args
        self.kwargs = kwargs

        # Build hybrid class of learning gent and policy agent
        python_commands = [
            f"global {self.policy_agent}{self.learning_agent}",
            f"class {self.policy_agent}{self.learning_agent}(agents.{self.policy_agent} ,agents.{self.learning_agent}):",
            "    def __init__(self, *args, **kwargs):",
            "        super().__init__(*args, **kwargs)"
        ]
        exec("\n".join(python_commands))

        self.is_set = True

    def reset(self):
        """
        Reset the agent allows new agent types to be added to build a new type of agent.
        """
        self.learning_agent = "NullLearning"
        self.policy_agent = "NullPolicy"
        self.is_set = False

    def make(self) -> Agent:
        """
        Constructs the new agent. This may be called over and over without having to pass in constructor arguments
        as long as they have been set.
        :return: The constructed agent
        """
        if not self.is_set:
            self.set()

        python_commands = [
            "global agent",
            f"agent = {self.policy_agent}{self.learning_agent}(*self.args, **self.kwargs)"
        ]
        try:
            exec("\n".join(python_commands))
        except NameError:
            logging.error("Must reset builder before making new type of agent.")
        return agent


if __name__ == "__main__":
    builder = AgentBuilder(policy="EGreedy", learning="TemporalDifferenceZero")
    builder.set(exploratory_rate=0.1, learning_rate=0.5, discount_rate=-1.0)
    td_agent = builder.make()
    td_agent2 = builder.make()
    builder.reset()

    null_agent = builder.make()
    builder.reset()
    assert null_agent.__class__.__name__ == "NullPolicyNullLearning"

    builder.add(agent_type="Random")
    random_agent = builder.make()
    builder.reset()
    assert random_agent.__class__.__name__ == "RandomNullLearning"

    builder.add(agent_type="Random")
    builder.add(agent_type="TemporalDifferenceZero")
    builder.set(discount_rate=1.0, learning_rate=lambda n: 1 / n)
    agent = builder.make()
    assert agent.__class__.__name__ == "RandomTemporalDifferenceZero"
