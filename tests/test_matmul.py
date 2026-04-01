"""Tests for matrix multiplication comparing pysiclib against numpy."""

import numpy as np
from pysiclib import linalg


class TestMatmul:
    """Tests for Tensor.matmul() against np.matmul()."""

    def test_2x2_matmul(self) -> None:
        """Verify 2x2 matrix multiplication."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]])
        a = linalg.Tensor(a_np)
        b = linalg.Tensor(b_np)
        result = a.matmul(b)
        np.testing.assert_array_almost_equal(result.to_numpy(), np.matmul(a_np, b_np))

    def test_3x3_matmul(self) -> None:
        """Verify 3x3 matrix multiplication."""
        a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        b_np = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
        a = linalg.Tensor(a_np)
        b = linalg.Tensor(b_np)
        result = a.matmul(b)
        np.testing.assert_array_almost_equal(result.to_numpy(), np.matmul(a_np, b_np))

    def test_non_square_matmul(self) -> None:
        """Verify non-square matrix multiplication (3x2 @ 2x4)."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b_np = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        a = linalg.Tensor(a_np)
        b = linalg.Tensor(b_np)
        result = a.matmul(b)
        np.testing.assert_array_almost_equal(result.to_numpy(), np.matmul(a_np, b_np))

    def test_vector_matmul(self) -> None:
        """Verify matrix-vector multiplication (2x3 @ 3x1)."""
        a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b_np = np.array([[1.0], [2.0], [3.0]])
        a = linalg.Tensor(a_np)
        b = linalg.Tensor(b_np)
        result = a.matmul(b)
        np.testing.assert_array_almost_equal(result.to_numpy(), np.matmul(a_np, b_np))

    def test_identity_matmul(self) -> None:
        """Verify multiplication by identity matrix returns the original."""
        a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        eye_np = np.eye(3)
        a = linalg.Tensor(a_np)
        eye = linalg.Tensor(eye_np)
        result = a.matmul(eye)
        np.testing.assert_array_almost_equal(result.to_numpy(), a_np)
