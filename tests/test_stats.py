"""Tests for the stats module comparing against numpy equivalents."""

import numpy as np
from pysiclib import linalg, stats


class TestStats:
    """Tests for find_mean, find_variance, find_stddev, find_skew, find_kurtosis."""

    def test_find_mean(self) -> None:
        """Verify find_mean matches numpy mean."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        t = linalg.Tensor(data)
        result = stats.find_mean(t, 0).get_buffer()[0]
        assert abs(result - np.mean(data)) < 1e-10

    def test_find_variance(self) -> None:
        """Verify find_variance matches numpy population variance."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        t = linalg.Tensor(data)
        result = stats.find_variance(t, 0).get_buffer()[0]
        assert abs(result - np.var(data)) < 1e-10

    def test_find_stddev(self) -> None:
        """Verify find_stddev matches numpy population std."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        t = linalg.Tensor(data)
        result = stats.find_stddev(t, 0).get_buffer()[0]
        assert abs(result - np.std(data)) < 1e-10

    def test_find_skew_symmetric(self) -> None:
        """Verify skew of symmetric data is zero."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        t = linalg.Tensor(data)
        result = stats.find_skew(t, 0).get_buffer()[0]
        assert abs(result) < 1e-10

    def test_find_skew_asymmetric(self) -> None:
        """Verify skew of asymmetric data is non-zero."""
        data = [1.0, 1.0, 1.0, 1.0, 10.0]
        t = linalg.Tensor(data)
        result = stats.find_skew(t, 0).get_buffer()[0]
        assert result > 0  # right-skewed

    def test_find_kurtosis_symmetric(self) -> None:
        """Verify kurtosis is computed and non-negative for symmetric data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        t = linalg.Tensor(data)
        result = stats.find_kurtosis(t, 0).get_buffer()[0]
        # siclib computes sum of standardized 4th central moments
        # For [1,2,3,4,5]: sum((x-mean)^4/n / sqrt(var)) = 4.808...
        n = len(data)
        mean = np.mean(data)
        var = np.var(data)
        expected = sum(((x - mean) ** 4 / n) / np.sqrt(var) for x in data)
        assert abs(result - expected) < 1e-10

    def test_find_mean_2d(self) -> None:
        """Verify find_mean on a 2D tensor along dim 0."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = linalg.Tensor(data)
        result = stats.find_mean(t, 0).squeeze().to_numpy()
        np.testing.assert_array_almost_equal(result, np.mean(data, axis=0))
