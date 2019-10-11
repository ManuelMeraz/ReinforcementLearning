#! /usr/bin/env python3
from rl.agents.agent import Agent
from rl.agents.learning import LearningAgent
from rl.agents.policy import PolicyAgent


class AgentBuilder:

    def __init__(self, policy_agent: str = None, learning_agent: str = None):
        learning_agents = LearningAgent.__subclasses__()
        policy_agents = PolicyAgent.__subclasses__()

        self.learning_agent_names = [agent.__name__ for agent in learning_agents]
        self.policy_agent_names = [agent.__name__ for agent in policy_agents]

        agent_subclasses = learning_agents + policy_agents
        agent_subclass_names = [agent.__name__ for agent in agent_subclasses]
        self.registry = dict(zip(agent_subclass_names, agent_subclasses))

        self.agents = []
        if policy_agent is not None:
            self.agents.append(policy_agent)

        if learning_agent is not None:
            self.agents.append(learning_agent)

    def add(self, agent_type: str):
        assert agent_type in self.registry, "Invalid agent type"
        self.agents.append(agent_type)
        assert len(self.agents) <= 2, "Can't add more than 2 agents"

    def make(self, *args, **kwargs) -> Agent:
        assert len(self.agents) == 2, "Must add in agents before making"
        first_agent = self.registry[self.agents.pop()]
        second_agent = self.registry[self.agents.pop()]
        if first_agent.__name__ in self.learning_agent_names:
            learning_agent = first_agent
            policy_agent = second_agent
        else:
            learning_agent = second_agent
            policy_agent = first_agent

        assert learning_agent.__name__ in self.learning_agent_names, "One of the agent types must be a learning agent type"
        assert policy_agent.__name__ in self.policy_agent_names, "One of the agent types must be a policy agent type"
        exec(f"""
class {policy_agent.__name__}{learning_agent.__name__}(policy.{policy_agent.__name__}, learning.{learning_agent.__name__}):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

global agent
agent = {policy_agent.__name__}{learning_agent.__name__}(*args, **kwargs) """)

        return agent


if __name__ == "__main__":
    builder = AgentBuilder("NullPolicyAgent", "NullLearningAgent")
    agent = builder.make()
