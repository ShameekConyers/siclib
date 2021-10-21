
#pragma once
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <utility>
pybind11::array_t<double> initial_value_problem(
	std::function <pybind11::array_t<double>(
		double, pybind11::array_t<double>)>& system_of_eqs,
	pybind11::array_t<double> initial_conditions,
	double target_value,
	double initial_value
);
