import pprint


class Value:
    def __init__(self, value: float = 0.0, count: float = 0.0):
        """
        Stores the value of a state and how many times the agent has been in this state
        :param value: The total accumulated reward computed using temporal difference
        :param count:  The number of times the agent has been in this state.
                       May be fractional if merged with another agent.
        """
        self.value = value
        self.count = count

    def __str__(self) -> str:
        return pprint.pformat({"value": self.value, "count": self.count})

    def __repr__(self):
        return self.__str__()
