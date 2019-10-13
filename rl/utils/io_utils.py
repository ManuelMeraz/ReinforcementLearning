#! /usr/bin/env python3
import json
from collections import defaultdict, Counter

from rl.reprs import Value


def load_learning_agent(filename: str):
    """
    Load data from csv file and convert it to state values
    :param filename: The name of a csv file containing the data
    :return: The state value mapping
    """

    state_values = defaultdict(Value)
    transitions = defaultdict(Counter)
    with open(filename, "r") as f:
        data = json.load(f)
        state_values_data = data["state_values"]
        transitions_data = data["transitions"]

    for state_value in state_values_data:
        state_values[tuple(state_value[0])] = Value(**state_value[1])

    for state_action_counts in transitions_data:
        state_action_pair = tuple(state_action_counts[0])

        for state_count in state_action_counts[1]:
            counts = Counter()
            counts[tuple(state_count[0])] = state_count[1]
            transitions[state_action_pair] += counts

    return state_values, transitions


def save_learning_agent(agent, filename: str):
    """
    Save the state values into a csv file
    :param state_values: The state value mapping
    :param filename: The name of the file to write to
    """

    data = {"state_values": [], "transitions": []}

    for state, value in agent.state_values.items():
        if value.count > 0:
            data["state_values"].append([[float(num) for num in state], value.__dict__])

    for state_action, transition_counts in agent.transitions.items():
        transition = [[float(num) for num in state_action], []]
        for state, count in transition_counts.items():
            state_counts = [[float(num) for num in state], count]
            transition[1].append(state_counts)

        data["transitions"].append(transition)

    with open(filename, "w") as f:
        json.dump(data, f, sort_keys=True, indent=2)