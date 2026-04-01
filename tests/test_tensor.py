"""Tests for tensor construction, view semantics, squeeze/unsqueeze."""

import numpy as np
import pytest
from pysiclib import linalg


class TestTensorConstruction:
    """Tests for creating tensors from lists and numpy arrays."""

    def test_create_from_flat_list(self) -> None:
        """Verify tensor creation from a flat list."""
        t = linalg.Tensor([1.0, 2.0, 3.0])
        assert t.get_shape() == [3]
        assert t.get_buffer() == [1.0, 2.0, 3.0]

    def test_create_from_list_with_shape(self) -> None:
        """Verify tensor creation from a flat list with explicit shape."""
        t = linalg.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        assert t.get_shape() == [2, 3]

    def test_create_from_numpy_1d(self) -> None:
        """Verify tensor creation from a 1D numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        t = linalg.Tensor(arr)
        assert t.get_shape() == [3]
        np.testing.assert_array_equal(t.to_numpy(), arr)

    def test_create_from_numpy_2d(self) -> None:
        """Verify tensor creation from a 2D numpy array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = linalg.Tensor(arr)
        assert t.get_shape() == [2, 2]
        np.testing.assert_array_equal(t.to_numpy(), arr)

    def test_create_from_numpy_3d(self) -> None:
        """Verify tensor creation from a 3D numpy array."""
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        t = linalg.Tensor(arr)
        assert t.get_shape() == [2, 2, 2]
        np.testing.assert_array_equal(t.to_numpy(), arr)

    def test_shape_stride_offset(self) -> None:
        """Verify shape, stride, and offset for a standard 2D tensor."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = linalg.Tensor(arr)
        assert t.get_shape() == [2, 3]
        assert t.get_stride() == [3, 1]
        assert t.get_offset() == 0


class TestViewSemantics:
    """Tests for view semantics: transpose shares underlying buffer."""

    def test_transpose_shares_buffer(self) -> None:
        """Verify that transposing a tensor returns a view over the same data."""
        t = linalg.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        t_transposed = t.transpose()
        assert t.view_buffer() == t_transposed.view_buffer()
        assert t.get_offset() == t_transposed.get_offset()

    def test_transpose_changes_shape_and_stride(self) -> None:
        """Verify that transpose swaps shape and stride."""
        t = linalg.Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        t_t = t.transpose()
        assert t_t.get_shape() == [3, 2]
        assert t_t.get_stride() == [1, 3]

    def test_deep_copy_creates_independent_data(self) -> None:
        """Verify that deep_copy creates an independent copy with same values."""
        t = linalg.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        t_copy = t.deep_copy()
        np.testing.assert_array_equal(t.to_numpy(), t_copy.to_numpy())


class TestSqueezeUnsqueeze:
    """Tests for squeeze and unsqueeze operations."""

    def test_squeeze_removes_size_one_dims(self) -> None:
        """Verify squeeze removes all size-1 dimensions."""
        t = linalg.Tensor(np.array([[1.0, 2.0, 3.0]]))
        assert t.get_shape() == [1, 3]
        squeezed = t.squeeze()
        assert squeezed.get_shape() == [3]

    def test_squeeze_specific_dim(self) -> None:
        """Verify squeeze on a specific dimension."""
        t = linalg.Tensor(np.array([[[1.0, 2.0, 3.0]]]))
        assert t.get_shape() == [1, 1, 3]
        squeezed = t.squeeze(0)
        assert squeezed.get_shape() == [1, 3]

    def test_squeeze_preserves_view(self) -> None:
        """Verify squeeze returns a view over the same data."""
        t = linalg.Tensor(np.array([[1.0, 2.0, 3.0]]))
        squeezed = t.squeeze()
        assert t.view_buffer() == squeezed.view_buffer()

    def test_squeeze_non_one_dim_raises(self) -> None:
        """Verify squeeze raises when targeting a non-size-1 dimension."""
        t = linalg.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(RuntimeError):
            t.squeeze(0)

    def test_unsqueeze_adds_dim(self) -> None:
        """Verify unsqueeze adds a size-1 dimension at the given position."""
        t = linalg.Tensor([1.0, 2.0, 3.0])
        unsqueezed = t.unsqueeze(0)
        assert unsqueezed.get_shape() == [1, 3]

    def test_unsqueeze_preserves_view(self) -> None:
        """Verify unsqueeze returns a view over the same data."""
        t = linalg.Tensor([1.0, 2.0, 3.0])
        unsqueezed = t.unsqueeze(0)
        assert t.view_buffer() == unsqueezed.view_buffer()
