from setuptools import setup, find_packages
import os
from os import path
from dotenv import load_dotenv

# Model package name
NAME = 'felix_ml_tools'

# Dependecies for the package
with open('requirements.txt') as r:
    DEPENDENCIES = [
        dep for dep in map(str.strip, r.readlines())
        if all([not dep.startswith("#"),
                not dep.endswith("#dev"),
                len(dep) > 0])
    ]

# Project descrpition
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    version='0.0.1',
    description='Macgyver lib to resolver any machine learning problem.',
    long_description=LONG_DESCRIPTION,
    author='Fernando Felix',
    packages=find_packages(exclude=("tests", "docs")),
    # external packages as dependencies
    install_requires=DEPENDENCIES
)
