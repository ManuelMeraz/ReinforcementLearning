#! /usr/bin/env python3
import pickle


def load_learning_agent(filename: str):
    """
    Load data from csv file and convert it to state values
    :param filename: The name of a csv file containing the data
    :return: The state value mapping
    """

    with open(filename, "rb") as f:
        data = pickle.load(f)

    return data["state_values"], data["transitions"]


def save_learning_agent(agent, filename: str):
    """
    Save the state values into a csv file
    :param state_values: The state value mapping
    :param filename: The name of the file to write to
    """

    data = {
        "state_values": agent.state_values,
        "transitions": agent.transitions
    }

    with open(filename, "ab") as f:
        pickle.dump(data, f)
