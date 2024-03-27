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

1. In a terminal, activate the virtual environment provided by Poetry:

`$ poetry shell`

2. Install required dependencies specified in `pyproject.toml`:

`$ poetry install`

3. Execute the following command to test that your installation has worked:

`$ python3 example.py`

4. To exit the virtual environment, simply use:

`$ exit`

For more detailed information on Poetry, refer to the [official documentation](https://python-poetry.org/docs/basic-usage/).

### Usage

1. You can choose to work on CPU, Apple's MPS or GPU (Cuda).
Ensure that you have initialized the `device` variable properly for your specific setup and moved the model and input data to the appropriate device if you choose to use GPU acceleration.

```python
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
```

2. Import the necessary modules:

```python
import torch.nn as nn
import nn.symmequiv as symmequiv
```

3. Define a class using `SymmetricGrpEquivLinear`:

```python
 class SymmetricGrpEquivNN(nn.Module):
  def __init__(self, dim_n):
    super().__init__()
    self.n = dim_n
    self.layer1 = nn.Sequential(
        symmequiv.SymmetricGrpEquivLinear(dim_n = self.n, order_k = 3, order_l = 2).to(device),
        nn.ReLU(),
        symmequiv.SymmetricGrpEquivLinear(dim_n = self.n, order_k = 2, order_l = 2).to(device),
        nn.ReLU(),
        symmequiv.SymmetricGrpEquivLinear(dim_n = self.n, order_k = 2, order_l = 0).to(device)
    )

  def forward(self, x):
    x = self.layer1(x)
    return x
```

4. Define an instance of SymmetricGrpEquivNN, specifying the dimension dim_n:

```python
model = SymmetricGrpEquivNN(dim_n=5)
```

5. Proceed to train and test as you would with any PyTorch NN model.

## Tests

To run all tests, navigate to the root directory of the repository and execute:

`$ python3 -m unittest discover`

## License

This project is licensed under the [MIT License](LICENSE).


