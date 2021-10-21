#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// uses five point midpoint
double derivative_at_index(
	const pybind11::array_t<double> data,
	int target_index
);

double derivative_at_value(
	const pybind11::array_t<double> data,
	double value
);
