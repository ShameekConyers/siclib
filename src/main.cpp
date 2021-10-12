#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "./differentiation.hpp"


PYBIND11_MODULE(impl, m)
{
	m.def("_diff", &find_derivative_from_index);

}
