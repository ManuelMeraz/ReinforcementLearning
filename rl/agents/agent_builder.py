#! /usr/bin/env python3

from rl import agents


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
        policy_name = self.registry[self.policy_agent].__name__
        learning_name = self.registry[self.learning_agent].__name__

        exec(f"""
global {policy_name}{learning_name} 
class {policy_name}{learning_name}(agents.{policy_name} ,agents.{learning_name}):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """)

    def make(self) -> agents.Agent:
        policy_name = self.registry[self.policy_agent].__name__
        learning_name = self.registry[self.learning_agent].__name__

        exec(f"""
global agent
agent = {policy_name}{learning_name}(*self.args, **self.kwargs) """)

        return agent


if __name__ == "__main__":
    builder = AgentBuilder(policy="EGreedy", learning="TemporalDifference")
    builder.set(exploratory_rate=0.1, learning_rate=0.5)
    agent = builder.make()
