#include "initial_value.hpp"
#include <vector>
#include <exception>
#include <iostream>

pybind11::array_t<double> initial_value_problem(std::function <pybind11::array_t<double>(
	double, pybind11::array_t<double>)>& system_of_eqs,
	pybind11::array_t<double> initial_conditions,
	double target_value,
	double initial_value)
{
	double cur_step = initial_value;
	double step_size;
	if (abs(target_value - initial_value) > 1.0) {
		step_size = (target_value - initial_value) / 1000.0;
		// step_size = 0.2;
	}
	else {
		step_size = (target_value - initial_value) / 100.0;
		// step_size = 0.2;
	}

	// decode our inital condition vector
	pybind11::buffer_info initial_conditions_info = initial_conditions.request();
	double* initial_conditions_buffer = (double*)initial_conditions_info.ptr;
	size_t initial_conditions_size = initial_conditions_info.shape.at(0);

	// We initialize the approximation and k vector
	pybind11::array_t<double> approximations(initial_conditions_info);
	pybind11::array_t<double> approximations_tmp(initial_conditions_info);
	double* approx_buf = (double*)approximations.request().ptr;
	double* approx_tmp_buf = (double*)approximations_tmp.request().ptr;

	std::vector<std::vector<double>> k(
		4, std::vector<double>(initial_conditions_size)
	);

	// Safety Check
	{
		pybind11::array_t<double> func_result
			= system_of_eqs(cur_step, approximations_tmp);
		pybind11::buffer_info func_info = func_result.request();
		size_t func_result_shape = func_info.shape.at(0);
		assert(
			func_info.shape == initial_conditions_info.shape
			&& func_info.shape.size() == 1
		);
	}

	// Algorithm Driver
	for (; cur_step < target_value - step_size / 2.0; cur_step += step_size) {

		pybind11::array_t<double> func_result
			= system_of_eqs(cur_step, approximations_tmp);
		double* func_buf = (double*)func_result.request().ptr;
		for (size_t j = 0; j < initial_conditions_size; j++) {
			k[0][j] = step_size * func_buf[j];
			// set up for k[1]
			approx_tmp_buf[j] = approx_buf[j] + 0.5 * k[0][j];
		}

		func_result
			= system_of_eqs(cur_step + step_size / 2.0, approximations_tmp);
		func_buf = (double*)func_result.request().ptr;
		for (size_t j = 0; j < initial_conditions_size; j++) {
			k[1][j] = step_size * func_buf[j];
			// set up for k[2]
			approx_tmp_buf[j] = approx_buf[j] + 0.5 * k[1][j];
		}

		func_result
			= system_of_eqs(cur_step + step_size / 2.0, approximations_tmp);
		func_buf = (double*)func_result.request().ptr;
		for (size_t j = 0; j < initial_conditions_size; j++) {
			k[2][j] = step_size * func_buf[j];
			// set up for k[3]
			approx_tmp_buf[j] = approx_buf[j] + k[2][j];
		}

		func_result
			= system_of_eqs(cur_step + step_size, approximations_tmp);
		func_buf = (double*)func_result.request().ptr;
		for (size_t j = 0; j < initial_conditions_size; j++) {
			k[3][j] = step_size * func_buf[j];
			// set up for next k[0]
			approx_buf[j] = approx_buf[j]
				+ (k[0][j] + 2 * k[1][j] + 2 * k[2][j] + k[3][j]) / 6;
			approx_tmp_buf[j] = approx_buf[j];
		}
		// std::cout << "\n";
		// std::cout << cur_step << "\t" << approx_buf[0] << "\t";
		// for (auto v : k) {
		// 	for (auto d : v) {
		// 		std::cout << d << ", ";
		// 	}
		// }
	}

	return approximations;
}
