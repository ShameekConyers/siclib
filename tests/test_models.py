"""Tests for the models module: LinearModel and KNearestNeighbors."""

from pysiclib import linalg, models


class TestLinearModel:
    """Tests for OLS regression."""

    def test_simple_linear_fit(self) -> None:
        """Verify OLS on simple linear data produces reasonable coefficients."""
        data = linalg.Tensor([[1.0, 2.0], [2.0, 4.0], [6.0, 5.0]])
        y = linalg.Tensor([2.5, 5.5, 8.7])

        model = models.LinearModel()
        model.fit_model(data, y)

        test_input = linalg.Tensor([4.0, 5.0])
        prediction = model.predict(test_input).get_item()
        # With an intercept, prediction should be in a reasonable range
        assert 5.0 < prediction < 12.0

    def test_perfect_linear_data(self) -> None:
        """Verify OLS recovers exact coefficients on noiseless data (y = 2*x1 + 3*x2)."""
        data = linalg.Tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ])
        # y = 2*x1 + 3*x2 (no intercept in the generating process)
        y = linalg.Tensor([2.0, 3.0, 5.0, 7.0])

        model = models.LinearModel()
        model.fit_model(data, y)

        test_input = linalg.Tensor([3.0, 2.0])
        prediction = model.predict(test_input).get_item()
        assert abs(prediction - 12.0) < 0.5


class TestKNearestNeighbors:
    """Tests for KNN classification."""

    def test_trivially_separable(self) -> None:
        """Verify KNN correctly classifies trivially separable data."""
        data = linalg.Tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [5.0, 2.0],
            [2.0, 4.0],
            [6.0, 5.0],
        ])
        labels = linalg.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        model = models.KNearestNeighbors()
        model.fit_model(data, labels, 2)

        near_class_0 = linalg.Tensor([0.5, 0.5])
        assert model.predict(near_class_0).squeeze().get_item() == 0.0

        near_class_1 = linalg.Tensor([4.0, 5.0])
        assert model.predict(near_class_1).squeeze().get_item() == 1.0

    def test_knn_boundary_point(self) -> None:
        """Verify KNN with k=1 assigns the label of the single nearest neighbor."""
        data = linalg.Tensor([
            [0.0, 0.0],
            [10.0, 10.0],
        ])
        labels = linalg.Tensor([0.0, 1.0])

        model = models.KNearestNeighbors()
        model.fit_model(data, labels, 1)

        close_to_first = linalg.Tensor([1.0, 1.0])
        assert model.predict(close_to_first).squeeze().get_item() == 0.0
