# Connecting Permutation Equivariant Neural Networks and Partition Diagrams

This repository contains an implementation based on the research 
presented in the paper by 
Pearce-Crump (2023): "Connecting Permutation Equivariant Neural Networks and Partition Diagrams". 
The core component provided here is the class `SymmetricGrpEquivLayer` within `nn/symmequiv.py`, which enables the creation of linear layer functions 
that exist in 
permutation equivariant neural networks where the layers 
are some tensor power of
\(\mathbb{R}^n\).

## Features

- PyTorch implementation of `SymmetricGrpEquivLayer` class facilitating the construction of linear layers for permutation equivariant neural networks.
- Flexible choice between the diagram basis and the orbit basis through the `diag` boolean flag.

## Installation and Usage

We recommend using Poetry, a dependency management and packaging tool for Python projects.

### Installation

1. Activate the virtual environment provided by Poetry:

`$ poetry shell`

2. Install required dependencies specified in `pyproject.toml`:

`$ poetry install`

3. Execute the following command to test that your installation has worked:

`$ python3 example.py`

4. To exit the virtual environment, simply use:

`$ exit$

For more detailed information on Poetry, refer to the [official documentation](https://python-poetry.org/docs/basic-usage/).

## Tests

To run all tests, navigate to the root directory of the repository and execute:

`$ python3 -m unittest discover`

## License

This project is licensed under the [MIT License](LICENSE).


