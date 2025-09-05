#!/usr/bin/env python
# encoding: utf-8

from setuptools import find_packages, setup

setup(
    name="clifs",
    version="0.1.0",
    description="cognitive linguistic identity fusion score",
    author="Devin R. Wright",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.10",
)
