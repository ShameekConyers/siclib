#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "./differentiation.hpp"


PYBIND11_MODULE(_sicnumerical, m)
{
	m.def("diff_from_index", &find_derivative_from_index);
	m.def("diff_from_value", &find_derivative_from_value);
}
