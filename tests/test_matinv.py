"""Tests for matrix inverse verifying A @ A_inv ≈ I."""

import numpy as np
from pysiclib import linalg


class TestMatinv:
    """Tests for Tensor.matinv()."""

    def test_2x2_inverse(self) -> None:
        """Verify A @ A_inv ≈ I for a 2x2 matrix."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = linalg.Tensor(a_np)
        a_inv = a.matinv()
        identity = a.matmul(a_inv)
        np.testing.assert_array_almost_equal(identity.to_numpy(), np.eye(2))

    def test_3x3_inverse(self) -> None:
        """Verify A @ A_inv ≈ I for a 3x3 matrix."""
        a_np = np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]])
        a = linalg.Tensor(a_np)
        a_inv = a.matinv()
        identity = a.matmul(a_inv)
        np.testing.assert_array_almost_equal(identity.to_numpy(), np.eye(3))

    def test_identity_is_its_own_inverse(self) -> None:
        """Verify the identity matrix is its own inverse."""
        eye_np = np.eye(3)
        eye = linalg.Tensor(eye_np)
        eye_inv = eye.matinv()
        np.testing.assert_array_almost_equal(eye_inv.to_numpy(), eye_np)

    def test_inverse_matches_numpy(self) -> None:
        """Verify matinv result matches np.linalg.inv."""
        a_np = np.array([[4.0, 7.0], [2.0, 6.0]])
        a = linalg.Tensor(a_np)
        a_inv = a.matinv()
        np.testing.assert_array_almost_equal(a_inv.to_numpy(), np.linalg.inv(a_np))
