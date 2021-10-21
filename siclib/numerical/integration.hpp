#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


// uses composite simpson's rule
double integral_index_interval(
	const pybind11::array_t<double> input,
	size_t start_index,
	size_t end_index
);
