#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


double find_derivative_from_index(
	const pybind11::array_t<double> data,
	int target_index
);

double find_derivative_from_value(
	const pybind11::array_t<double> data,
	double value
);
