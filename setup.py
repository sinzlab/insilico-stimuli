#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="insilico_stimuli",
    version="0.0.0",
    description="Insilico Experiments",
    author="Dominik Kessler",
    author_email="dominik.kessler@uni-oldenburg.de",
    packages=find_packages(exclude=[]),
    install_requires=['sphinx', 'pytorch_sphinx_theme', 'recommonmark'],
)