![CI](https://github.com/ShameekConyers/siclib/actions/workflows/ci.yml/badge.svg)

# siclib

A C++ scientific computation library with a focus on math. pysiclib is the Python interface.

## What This Demonstrates

- View-based N-dimensional tensor architecture with constant-time transpose
- BLAS integration for matrix operations (Apple Accelerate on macOS, OpenBLAS on Linux)
- Python interop via pybind11
- N-dimensional broadcasting for element-wise operations
- Functional composition of operations over tensor views

## Project and Documentation

Can be found <a href="https://shameekconyers.com/projects/siclib">here</a>.

A neural net demo using pysiclib can be found <a href="https://shameekconyers.com/projects/pysiclib_neuralnet_demo">here</a>.

## Modules

- **Linalg** - N-dimensional tensor with view semantics, matmul via BLAS, matrix inverse, transpose, squeeze/unsqueeze, slicing, and broadcasting
- **Stats** - Moment generating function, mean, variance, standard deviation, skew, kurtosis, and random tensor generation
- **Numerical** - Numerical differentiation, integration, equation solving, and initial value problem solver
- **Adaptive** - Neural network implementation (ProtoNet) built on top of the tensor library
- **Models** - OLS linear regression and K-nearest neighbors classification

## pysiclib - Python Installation

Requirements: a C++ compiler, CMake, and pybind11. On Linux you also need libopenblas-dev.

```shell
git clone https://github.com/ShameekConyers/siclib
cd siclib
python -m venv .venv
source .venv/bin/activate
pip install pybind11 numpy
pip install -e .
```

```python
>>> import pysiclib
>>> from pysiclib import linalg, stats

>>> my_matrix = [[[0, 1, 2], [3, 4, 5]],
...              [[6, 7, 8], [9, 10, 11]]]
>>> my_tensor = linalg.Tensor(my_matrix)
>>> print(my_tensor)

Tensor:
[[[0, 1, 2]
  [3, 4, 5]]
 [[6, 7, 8]
  [9, 10, 11]]]
Tensor Shape: [2, 2, 3]

>>> # Tensors are views, so transpose runs in constant time
>>> # and shares the underlying buffer
>>> transposed = my_tensor.transpose()
>>> print(transposed)

Tensor:
[[[0, 6]
  [3, 9]]
 [[1, 7]
  [4, 10]]
 [[2, 8]
  [5, 11]]]
Tensor Shape: [3, 2, 2]

>>> # Stats functions work over arbitrary dimensions
>>> col_means = stats.find_mean(transposed, 1)
>>> print(col_means)

Tensor:
[[[1.5, 2.5, 3.5]]
 [[7.5, 8.5, 9.5]]]
Tensor Shape: [2, 1, 3]
```

## Running Tests

```shell
source .venv/bin/activate
pip install pytest numpy scipy
pytest tests/ -v
```
