#! /usr/bin/env3
import cProfile
import functools
import io
import pstats
from pstats import SortKey


def profiled(function):
    """
    Decorator that calls profiles a function and prints out statistics
    Usage:

    @perf_tools.profiled
    def func():
        // code

    :param function: a python function
    :type function: python function
    """

    @functools.wraps(function)
    def profiled_function(*args, **kwargs):
        """
        Prints out debug information before and after function is called
        """
        profile = cProfile.Profile()
        profile.enable()
        return_value = function(*args, **kwargs)
        profile.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return return_value

    return profiled_function
