#! /usr/bin/env python3
import os
from collections import defaultdict
from typing import Dict, Tuple, Any

import pandas

from rl.agents.learning import Value


def load_state_values(filename: str) -> Dict[Tuple[Any], Value]:
    """
    Load data from csv file and convert it to state values
    :param filename: The name of a csv file containing the data
    :return: The state value mapping
    """
    if os.path.exists(filename):
        data = pandas.read_csv(filename)
        data = data.replace({pandas.np.nan: None}).values.tolist()
        state_values = defaultdict(Value)
        for d in data:
            state_values[tuple(d[1:11])] = Value(value=d[11], count=d[12])
        return state_values
    else:
        return defaultdict(Value)


def save_state_values(state_values: Dict[Tuple[Any], Value], filename: str):
    """
    Save the state values into a csv file
    :param state_values: The state value mapping
    :param filename: The name of the file to write to
    """
    data = []
    for key, value in state_values.items():
        if isinstance(key, tuple) and value.count > 0:
            data.append(key + (value.value, value.count))

    df = pandas.DataFrame(data)
    df.to_csv(filename)
