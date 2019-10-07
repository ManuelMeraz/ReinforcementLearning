#! /usr/bin/env python3
from collections import defaultdict
from typing import Dict

import pandas

from rl.agents.learning import Value


def load_state_values(filename: str) -> Dict[bytes, Value]:
    """
    Load data from csv file and convert it to state values
    :param filename: The name of a csv file containing the data
    :return: The state value mapping
    """
    data = pandas.read_csv(filename)
    data = data.values.tolist()
    state_values = defaultdict(Value)
    for d in data[1:]:
        state_values[d[1]] = Value(value=d[2], count=d[3])
    return state_values


def save_state_values(state_values: Dict[bytes, Value], filename: str):
    """
    Save the state values into a csv file
    :param state_values: The state value mapping
    :param filename: The name of the file to write to
    """
    data = []
    for key, value in state_values.items():
        if value.count > 0:
            data.append((key, value.value, value.count))

    df = pandas.DataFrame(data)
    df.to_csv(filename)
