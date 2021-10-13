#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "./differentiation.hpp"
#include "./integration.hpp"


PYBIND11_MODULE(_sicnumerical, m)
{
	m.def(
		"find_derivative_from_index", &find_derivative_from_index);
	m.def(
		"find_derivative_from_value", &find_derivative_from_value);
	m.def(
		"find_integral_from_index", &find_integral_from_index);
}
