#include "sol_equations.hpp"
#include <array>
#include <exception>
#include <cmath>

std::optional<double> equation_solution(
	const std::function<double(double)>& input_func,
	double target_val)
{

	double epsilon = 1e-10;
	size_t max_num_iter = 1e10;

	double cur = 0.0;
	double prev_1 = 1.26;
	double prev_2 = 0.23;
	double val_1 = input_func(prev_1) - target_val;
	double val_2 = input_func(prev_2) - target_val;

	for (size_t cur_iter = 0; cur_iter < max_num_iter; cur_iter++) {
		cur = prev_1 - val_1
			* (prev_1 - prev_2) / (val_1 - val_2);
		if (abs(cur - prev_1) < epsilon) {
			return cur;
		}

		prev_2 = prev_1;
		prev_1 = cur;
		val_2 = val_1;
		val_1 = input_func(prev_1) - target_val;
	};

	return {};
}
