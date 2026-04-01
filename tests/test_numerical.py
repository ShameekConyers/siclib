"""Tests for the numerical module: derivatives, integrals, IVP, equation solving."""

from pysiclib import linalg, numerical


class TestNumerical:
    """Tests for derivative, integral, equation solving, and IVP."""

    def test_derivative_of_quadratic(self) -> None:
        """Verify derivative of x^2 + x at index 5 equals 2*5 + 1 = 11."""
        data = linalg.Tensor([x ** 2 + x for x in range(11)])
        assert numerical.derivative_at_index(data, 5) == 11

    def test_integral_of_quadratic(self) -> None:
        """Verify integral approximation of x^2 + x over [0, 5]."""
        unit_steps = 100
        data = linalg.Tensor(
            [(x / unit_steps) ** 2 + (x / unit_steps)
             for x in range(11 * unit_steps)]
        )
        result = numerical.integral_index_interval(data, 0, 5 * unit_steps) / unit_steps
        # Analytical: integral of x^2 + x from 0 to 5 = 125/3 + 25/2 ≈ 54.17
        assert abs(result - 50) < 10

    def test_equation_solution_quadratic(self) -> None:
        """Verify finding root of x^2 + 2x - 1 starting near 14 gives x=3 (approx root of shifted)."""
        def f(x: float) -> float:
            """Compute x^2 + 2x - 1."""
            return x ** 2 + 2 * x - 1

        result = numerical.equation_solution(f, 14)
        assert result == 3.0

    def test_equation_solution_no_root(self) -> None:
        """Verify None is returned when no real root exists."""
        def f(x: float) -> float:
            """Compute x^2 + 1 (always positive)."""
            return x ** 2 + 1

        result = numerical.equation_solution(f, 0)
        assert result is None

    def test_initial_value_problem(self) -> None:
        """Verify IVP solver on a known system of ODEs."""
        def system(t: float, var_arr: linalg.Tensor) -> linalg.Tensor:
            """Compute derivatives for a 2-variable ODE system."""
            buf = var_arr.get_buffer()[:]
            d0 = -4 * buf[0] + 3 * buf[1] + 6
            d1 = 0.6 * d0 - 0.2 * buf[1]
            return linalg.Tensor([d0, d1])

        initial = linalg.Tensor([0.0, 0.0])
        result = numerical.initial_value_problem(system, initial, 0.5, 0.0)
        buf = result.get_buffer()
        assert abs(buf[0] - 1.79353) < 1e-4
        assert abs(buf[1] - 1.01442) < 1e-4
