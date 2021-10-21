#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "numerical.hpp"
#include "linalg.hpp"

PYBIND11_MODULE(_pysiclib, m)
{
	// numerical
	m.def(
		"derivative_at_index", &derivative_at_index);
	m.def(
		"derivative_at_value", &derivative_at_value);
	m.def(
		"integral_index_interval", &integral_index_interval);
	m.def(
		"equation_solution", &equation_solution
	);
	m.def(
		"initial_value_problem", &initial_value_problem
	);

	// linalg
}
