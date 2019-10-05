#!/usr/bin/env python3

import os

import setuptools

DIR = os.path.dirname(__file__)
REQUIREMENTS = os.path.join(DIR, "requirements.txt")

with open(REQUIREMENTS) as f:
    reqs = f.read()

setuptools.setup(
    name="rl",
    version="0.0.1",
    description="Reinforcement Learning: An Introduction",
    url="github.com/manuelmeraz/ReinforcementLearning",
    author="Manuel Meraz-Rodriguez",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=reqs.strip().split("\n"),
    entry_points={"console_scripts": ["tictactoe = rl.tictactoe.main:main"]},
)
