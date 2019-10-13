#! /usr/bin/env python3

from rl import agents
import logging


class AgentBuilder:

    def __init__(self, policy: str = "NullPolicy", learning: str = "NullLearning"):
        learning_agents = agents.LearningAgent.__subclasses__()
        policy_agents = agents.PolicyAgent.__subclasses__()

        self.learning_agent_names = [agent.__name__ for agent in learning_agents]
        self.policy_agent_names = [agent.__name__ for agent in policy_agents]

        agent_subclasses = learning_agents + policy_agents
        agent_subclass_names = [agent.__name__ for agent in agent_subclasses]
        self.registry = dict(zip(agent_subclass_names, agent_subclasses))

        self.learning_agent = learning
        self.policy_agent = policy

        # Constructors arguments
        self.args = []
        self.kwargs = {}
        self.is_set = False

    def add(self, agent_type: str):
        assert agent_type in self.registry, "Invalid agent type"
        assert self.learning_agent == "NullLearning" or self.policy_agent == "NullPolicy", \
            "Cannot add in more agent types."

        if agent_type in self.learning_agent_names:
            self.learning_agent = agent_type
        else:
            self.policy_agent = agent_type

    def set(self, *args, **kwargs):
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
        self.learning_agent = "NullLearning"
        self.policy_agent = "NullPolicy"
        self.is_set = False

    def make(self) -> agents.Agent:

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
    builder = AgentBuilder(policy="EGreedy", learning="TemporalDifference")
    builder.set(exploratory_rate=0.1, learning_rate=0.5)
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
    builder.add(agent_type="TemporalDifferenceAveraging")
    agent = builder.make()
    assert agent.__class__.__name__ == "RandomTemporalDifferenceAveraging"
