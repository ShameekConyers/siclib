#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


// uses composite simpson's rule
double find_integral_from_index(
	const pybind11::array_t<double> input,
	size_t start_index,
	size_t end_index
);
