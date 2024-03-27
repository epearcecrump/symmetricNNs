# Connecting Permutation Equivariant Neural Networks and Partition Diagrams

## Description

This repo contains the class 

nn/symmequiv.py: SymmetricGrpEquivLayer

that creates the linear layer functions that appear in a permutation
equivariant neural network where the layers are some tensor power of R^n.

The boolean flag diag allows you to choose either the diagram basis or the orbit basis when constructing the linear layers.

## Tests

To execute all tests, first return to the high-level directory symmetricNNs, and then run

$ python3 -m unittest discover
