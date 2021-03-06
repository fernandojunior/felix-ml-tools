from setuptools import setup, find_packages

# Model package name
NAME = "felix_ml_tools"

# Dependecies for the package
with open("requirements.txt") as r:
    DEPENDENCIES = [
        dep
        for dep in map(str.strip, r.readlines())
        if all([not dep.startswith("#"), len(dep) > 0])
    ]

# Project descrpition
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    version="0.0.1",
    description="A ML/EDA lib to rule any problem.",
    long_description=LONG_DESCRIPTION,
    author="Fernando Felix",
    packages=find_packages(exclude=("tests", "docs")),
    # external packages as dependencies
    install_requires=DEPENDENCIES,
)
