#! /usr/bin/env python3
# coding=utf-8
"""
Contains all logging related tools and logging settings.
"""

import argparse
import functools
import logging
import os
import pprint
import sys

# Convert verbose arguments to their corresponding level
_string_to_level = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def logged(function):
    """
    Decorator that calls logging.debug on the arguments and return value for the
    function it decorates.

    Usage:

    @logging_tools.logged
    def func():
        // code

    :param function: a python function
    :type function: python function
    """

    @functools.wraps(function)
    def logged_function(*args, **kwargs):
        """
        Prints out debug information before and after function is called
        """
        logging.debug(f"Begin function: {function.__name__}")
        logging.debug(f"Arguments:\n{pprint.pformat(args)}")
        logging.debug(f"Keyword arguments:\n{pprint.pformat(kwargs)}")
        return_value = function(*args, **kwargs)
        logging.debug(f"End function: {function.__name__}")
        logging.debug(
            f"Return type({type(return_value)}):\n{pprint.pformat(return_value)}"
        )

        return return_value

    return logged_function


class Logger:
    """
    Instantiating this class enables logging.

    Automatically adds a verbose option to your command line options.

    If using argparse.ArgumentParser() to parse arguments, then the user should pass in the parser as an argument to
    this class when instantiating it to add the options to your current command line options.
    """

    def __init__(
            self,
            filename=None,
            format="[%(levelname)s]: %(message)s",
            level=logging.WARNING,
            parser=argparse.ArgumentParser(),
    ):
        """
        :param filename: If a filename is passed in and the logging command line option is passed in, then the logger
                         will use that filename for the log file.
        :type filename: string

        :param format: The format for the log output. Please see the python logging documentation for more info.
        :type format: string

        :param level: The logging level corresponds to the logging levels: DEBUG, INFO, WARNING, ERROR.
                      The command line argument will override this.
        :type level: int

        :param parser: If using argparse, pass the argument parser to this class to add in the verbose flags.
        :type parser: argparse.ArgumentParser
        """

        self.filename = filename
        self.file_handle = None
        self.format = format
        self.level = level

        parser.add_argument(
            "-v",
            default=0,
            help="Print out additional details while the program is running.",
            action="count",
        )

        parser.add_argument(
            "--verbose",
            default=self.level,
            help="DEBUG, INFO, WARNING, ERROR. Default is WARNING.",
            metavar="LEVEL",
        )

        parser.add_argument(
            "--log",
            default=False,
            help="Write to log file at max verbosity level.",
            action="store_true",
        )

        options, _ = parser.parse_known_args()

        # not the default
        if options.verbose != self.level:
            try:
                options.verbose = _string_to_level[options.verbose.upper()]
            except (AttributeError, KeyError):
                logging.warning("Invalid logging level. Using WARNING")
                logging.warning("DEBUG, INFO, WARNING or ERROR are available.")
                options.verbose = logging.WARNING

        elif options.v:
            if options.v > 1:
                options.v = logging.DEBUG
            elif options.v > 0:
                options.v = logging.INFO

            options.verbose = options.v

        self.level = options.verbose

        if options.verbose >= logging.DEBUG or options.log:
            self.format = (
                "[%(levelname)s][%(filename)s:%(lineno)s][%(funcName)s]: %(message)s"
            )

        elif options.verbose == logging.INFO:
            self.format = "[%(levelname)s][%(funcName)s]: %(message)s"

        # logging and stdout/stderr have different handles with the python logging package
        handlers = []

        if options.log:
            self.level = logging.DEBUG

            if not filename:
                import time

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.filename = os.getcwd() + "/" + timestamp + ".log"

            print("Log filename: ", self.filename)
            # Redirect python package logging output to log file
            # Does not catch prints or stdout/stderr
            logging_file_handler = logging.FileHandler(filename=self.filename)
            handlers.append(logging_file_handler)

            # Redirect stdout/stderr to logfile
            self.file_handle = open(self.filename, "r+")
            sys.stdout = self.file_handle
            sys.stderr = self.file_handle

        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)

        logging.basicConfig(format=self.format,
                            level=self.level,
                            handlers=handlers)

    def __del__(self):
        if self.file_handle:
            self.file_handle.close()
