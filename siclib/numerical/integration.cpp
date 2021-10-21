#include "integration.hpp"
#include <exception>
#include <iostream>

double integral_index_interval(
	const pybind11::array_t<double> input,
	size_t start_index,
	size_t end_index)
{
	pybind11::buffer_info input_buffer_info = input.request();
	double* input_buffer = (double*)input_buffer_info.ptr;
	size_t input_size = input_buffer_info.shape.at(0);

	if (input_size < 2) {
		throw std::runtime_error(
			"Need at least 2 points"
		);
	}
	if (end_index <= start_index + 1
		|| end_index > input_size || start_index >= input_size) {
		throw std::runtime_error(
			"Invalid indices"
		);
	}

	double endpoints_index_sum = input_buffer[0]
		+ input_buffer[input_size - 1];
	double even_index_sum = 0;
	double odd_index_sum = 0;


	for (size_t cur_index = start_index + 1; cur_index < end_index; cur_index++) {
		if ((cur_index - start_index) % 2 == 0) {
			even_index_sum += input_buffer[cur_index];
		}
		else {
			odd_index_sum += input_buffer[cur_index];
		}
	}

	double estimated_integral = (endpoints_index_sum + 2 * even_index_sum + 4 * odd_index_sum) / 3;

	return estimated_integral;
}
