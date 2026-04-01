"""Tests for broadcasting binary operations between tensors of different shapes."""

import numpy as np
from pysiclib import linalg


def _add(a: float, b: float) -> float:
    """Add two scalars."""
    return a + b


def _mul(a: float, b: float) -> float:
    """Multiply two scalars."""
    return a * b


class TestBroadcasting:
    """Tests for binary element-wise ops with broadcasting."""

    def test_broadcast_row_vector(self) -> None:
        """Verify broadcasting a row vector across a 2D tensor."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_np = np.array([[10.0, 20.0]])
        x = linalg.Tensor(x_np)
        y = linalg.Tensor(y_np)
        result = x.binary_element_wise_op(y, _add)
        np.testing.assert_array_almost_equal(result.to_numpy(), x_np + y_np)

    def test_broadcast_col_vector(self) -> None:
        """Verify broadcasting a column vector across a 2D tensor."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_np = np.array([[10.0], [20.0]])
        x = linalg.Tensor(x_np)
        y = linalg.Tensor(y_np)
        result = x.binary_element_wise_op(y, _add)
        np.testing.assert_array_almost_equal(result.to_numpy(), x_np + y_np)

    def test_broadcast_multiply(self) -> None:
        """Verify broadcasting with multiplication."""
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_np = np.array([[2.0, 3.0, 4.0]])
        x = linalg.Tensor(x_np)
        y = linalg.Tensor(y_np)
        result = x.binary_element_wise_op(y, _mul)
        np.testing.assert_array_almost_equal(result.to_numpy(), x_np * y_np)

    def test_same_shape_no_broadcast(self) -> None:
        """Verify element-wise op between tensors of the same shape."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_np = np.array([[5.0, 6.0], [7.0, 8.0]])
        x = linalg.Tensor(x_np)
        y = linalg.Tensor(y_np)
        result = x.binary_element_wise_op(y, _add)
        np.testing.assert_array_almost_equal(result.to_numpy(), x_np + y_np)
