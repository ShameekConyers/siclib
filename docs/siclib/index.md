---
nav_name: SicLib
name: "siclib"
title: SicLib
date_added:
date_edited:
description:
---

## A Scientific Computation Library
SicLib is a collection of various things I have Implemented in C++ , with a focus being on math. pysiclib is the package interface that allows
use through Python.

## Navigation
The various modules can be navigated to the left

## Source Code & Installation
Source code can be found <a href=https://github.com/ShameekConyers/siclib> here </a>

---

## Project Scope and Purpose

Despite there being much better Implementations of similar concepts, the purpose
of this is for me to get a better lower level understanding on applied scientific
computation. and allowing me to implement algorithms/methods I come across. Hence this
is more akin to a research project instead of a serious production ready implementation.

## Implementation Notes

The Tensor is the general data structure of this Library. However, they are fairly unoptimized as they are implemented with using generalizations, functional programming and using Data Views. I recommend PyTorch or TorchLib instead for a
general Tensor Library.

The general idea is to work with a composition of pure functions and views to allow refactoring into an optimized Implementation later - this will allow operations which are "safe" to be made parallelized oon the CPU or GPU

I'm looking into SYCL since CUDA already dominates the space and SYCL is usable on a wider set of systems, including iGPU's. Hence I will eventually
move over to SYCL-BLAS instead of BLAS when I decide to include GPU operations.

---

## Neural Network Module Example

An Example utilization can be found <a href=https://shameekconyers.com/projects/pysiclib_neuralnet_demo> here </a>

## Example Functionality

```python
>>> import pysiclib
>>> my_matrix = [[[0, 1, 2],[ 3, 4, 5]],
... 	[[6, 7, 8],[ 9, 10, 11]]]

>>> # Tensors can be constructed by arbitrary python arrays
>>> my_tensor = pysiclib.linalg.Tensor(my_matrix)
>>> print(my_tensor)

Tensor:
[[[0, 1, 2]
  [3, 4, 5]]
 [[6, 7, 8]
  [9, 10, 11]]]
Tensor Shape: [2, 2, 3]

>>> # Note Tensors are views so we can do things like
>>> # transpose in constant time
>>> # Below we will print the address in memory to show this
>>> transposed_my_tensor = my_tensor.transpose()
>>> print(transposed_my_tensor)

Tensor:
[[[0, 6]
  [3, 9]]
 [[1, 7]
  [4, 10]]
 [[2, 8]
  [5, 11]]]
Tensor Shape: [3, 2, 2]

>>> print(hex(id(my_tensor.get_buffer()))

0x7fcfe02891c0

>>> print(hex(id(transposed_my_tensor.get_buffer()))

0x7fcfe02891c0

>>> # Example of extensibility for generalized statistics
>>> # as mean is a convenience function over the
>>> # Moment Generating Function.
>>> t_col_means = pysiclib.stats.find_mean(
... 	transposed_my_tensor, 1)
>>> print(t_col_means)

Tensor:
[[[1.5, 2.5, 3.5]]
 [[7.5, 8.5, 9.5]]]
Tensor Shape: [2, 1, 3]




```
