# Connecting Permutation Equivariant Neural Networks and Partition Diagrams

## Description

This repo contains the class 

nn/symmequiv.py: SymmetricGrpEquivLayer

that creates the linear layer functions that appear in a permutation
equivariant neural network where the layers are some tensor power of R^n.

The boolean flag diag allows you to choose either the diagram basis or the orbit basis when constructing the linear layers.

## Using Poetry to execute the code

We recommend that you use Poetry, a dependency management and packaging tool in Python.

To run this, execute the following commands:

$ poetry shell (to activate the virtual environment that Poetry provides)

$ poetry install (to install the required dependencies that are declared in the pyproject.toml file)

$ python3 example.py (to see that this has worked)

To exit your new shell/virtual environment, use

$ exit

To find out more about Poetry, please view https://python-poetry.org/docs/basic-usage/


## Tests

To execute all tests, first return to the high-level directory symmetricNNs, and then run

$ python3 -m unittest discover
