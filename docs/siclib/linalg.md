---
nav_name: Linear Algebra
name: "linalg"
title:  siclib.linalg
date_added:
date_edited:
description:
---

## A Module based on Linear Algebra.

## Implementation Notes

Tensor operations are unoptimized. I recommend PyTorch or TorchLib instead for a
general Tensor Library.

---

## Theoretical Motivation
Given the standard ring of $(\R, +, *)$ we can represent all potential
operations as a composition of binary operators $f(x, y) = x * y$ or
$f(x, y) = x + y$ where $x, y \in R$. Given a vector space of $\R^n$ we can
represent a vector as $\textbf{x} = \{(x_0, ..., x_n): x_i \in \R\}$, where
$f(\textbf{x}, \textbf{y})$ is some combination of $f(x_i, y_j)$.
Given a $p$ collection of vectors of size $n$ We define a matrix
$\textbf{X} = (\textbf{x}_1, ..., \textbf{x}_p) \in \R^{n \times p}$
. We describe the structure of a Tensor-space $\mathfrak{T}$ of order
$m$ on $\R$ as

$$
\Large \mathfrak{T_\R^m} =\normalsize\prod_{i=1}^m \R^k,\enspace k \in \N^+
$$

Given the usual ring $(\R, +, *)$ we can extend many theorems of
linear algebra by allowing the following rule.

Given the tensors $\textbf{T}_1 = (\textbf{X}_1, ..., \textbf{X}_k)$,
$\textbf{T}_2 = (\textbf{Y}_1, ..., \textbf{Y}_p)$, and $\textbf{T}_3 =
(\textbf{Z}_1, ..., \textbf{Z}_q)$ a necessary (but not sufficient) condition
for the operation $f(\textbf{T}_1, \textbf{T}_2) = \textbf{T}_3$  to be well
defined is for
$0 < i \leq max(p, k),  \enspace dim(X_i) = dim(Y_i)$, $dim(X_i) = 1$, or
$dim(Y_i) = 1$ and $dim(Z_i) = max(p, k)$. Where WLOG
$dim(X_i) = 1$ if $k < p$.



---

## Documentation

```python
# pysiclib.linalg.Tensor
class Tensor:
    @overload
    def __init__(self, numpy_array: numpy.ndarray[numpy.float64]) -> None: ...
    @overload
    def __init__(self, input_data: List[float], input_shape: List[int] = ..., input_stride: List[int] = ..., offset: int = ...) -> None: ...
    @overload
    def __init__(self, other_view: Tensor) -> None: ...
    def binary_element_wise_op(self, arg0: Tensor, arg1: Callable[[float,float],float]) -> Tensor: ...
    def deep_copy(self) -> Tensor: ...
    def fold_op(self, arg0: Callable[[float,float],float], arg1: float, arg2: int, arg3: bool) -> Tensor: ...
    def get_buffer(self) -> List[float]: ...
    def get_offset(self) -> int: ...
    def get_shape(self) -> List[int]: ...
    def get_stride(self) -> List[int]: ...
    def to_numpy(self) -> numpy.ndarray[numpy.float64]: ...
    def transpose(self) -> Tensor: ...



```
