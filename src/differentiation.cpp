#include "./differentiation.hpp"
#include <exception>
#include <iostream>

double find_derivative_from_index(const pybind11::array_t<double> data,
	int target_index)
{
	pybind11::buffer_info buffer_data = data.request();
	double* buffer_ptr = (double*)buffer_data.ptr;
	int max_size = buffer_data.shape.at(0);
	double est_deriv;

	// boundary checks
	if (target_index < 0 || target_index >= max_size) {
		throw std::runtime_error(
			"each point needs at least one neighbor on each side");
	}


	// do five-point midpoint
	else if (target_index > 1 && target_index + 2 < max_size) {
		int i = target_index;
		est_deriv = (1.0 / 12.0)
			* (
				buffer_ptr[i - 2] - 8 * buffer_ptr[i - 1] + 8 * buffer_ptr[i + 1]
				- buffer_ptr[i + 2]);
	}
	// do three-point midpoint
	else if (target_index >= 1 && target_index + 2 <= max_size) {
		est_deriv = (1.0 / 2.0)
			* (-buffer_ptr[target_index - 1] + buffer_ptr[target_index + 1]);
	}
	// do a forward guess to keep bounds
	else if (target_index == 0) {
		est_deriv = buffer_ptr[1] - buffer_ptr[0];
	}
	// do a backward guess to keep bounds
	else {
		est_deriv = buffer_ptr[max_size - 1] - buffer_ptr[max_size - 2];
	}
	// std::cout << buffer_ptr[max_size - 1] << "\n";
	// std::cout << buffer_ptr[max_size - 2];

	return est_deriv;
}
