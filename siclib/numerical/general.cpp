
#include "general.hpp"
#include <array>
#include <exception>
#include <cmath>

namespace sic
{
std::optional<double> equation_solution(
	const std::function<double(double)>& input_func,
	double target_val)
{

	double epsilon = 1e-7;
	size_t max_num_iter = 1e3;

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


double integral_index_interval(
	const TensorView& input,
	size_t start_index,
	size_t end_index)
{
	std::vector<size_t> input_shape = input.get_shape();
	std::vector<size_t> input_stride = input.get_stride();
	size_t input_size = input_shape.at(0);

	if (input_shape.size() > 1 || input_stride[0] != 1) {
		throw std::runtime_error(
			"Squeeze the tensor such that it is 1 dimension and dense"
		);
	}

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

	auto input_buffer = input.m_storage->m_data; // compatability

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

TensorView
initial_value_problem(std::function <TensorView(double, TensorView)>& system_of_eqs,
	const TensorView& initial_conditions,
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
	std::vector<size_t> init_conditions_shape = initial_conditions.get_shape();
	std::vector<size_t> init_conditions_stride =
		initial_conditions.get_stride();

	if (init_conditions_shape.size() > 1 || init_conditions_stride[0] != 1) {
		throw std::runtime_error(
			"Squeeze the tensor such that it is 1 dimension and dense"
		);
	}
	size_t init_conditions_size = init_conditions_shape[0];


	// We initialize the approximation and k vector
	TensorView approximations = initial_conditions.deep_copy();
	TensorView approximations_tmp = initial_conditions.deep_copy();

	std::vector<double>& approx_buf = approximations.get_buffer();
	std::vector<double>& approx_tmp_buf = approximations_tmp.get_buffer();

	std::vector<std::vector<double>> k(
		4, std::vector<double>(init_conditions_size)
	);

	// Safety Check
	{
		TensorView func_result
			= system_of_eqs(cur_step, approximations_tmp);

		std::vector<size_t> func_result_shape = func_result.get_shape();
		size_t func_result_dim = func_result.get_shape().at(0);
		assert(
			func_result_dim == init_conditions_size
			&& func_result_shape.size() == 1
		);
	}

	// Algorithm Driver
	for (; cur_step < target_value - step_size / 2.0; cur_step += step_size) {

		TensorView func_result
			= system_of_eqs(cur_step, approximations_tmp);
		std::vector<double> func_buf = func_result.get_buffer();
		for (size_t j = 0; j < init_conditions_size; j++) {
			k[0][j] = step_size * func_buf[j];
			// set up for k[1]
			approx_tmp_buf[j] = approx_buf[j] + 0.5 * k[0][j];
		}

		func_result
			= system_of_eqs(cur_step + step_size / 2.0, approximations_tmp);
		func_buf = func_result.get_buffer();
		for (size_t j = 0; j < init_conditions_size; j++) {
			k[1][j] = step_size * func_buf[j];
			// set up for k[2]
			approx_tmp_buf[j] = approx_buf[j] + 0.5 * k[1][j];
		}

		func_result
			= system_of_eqs(cur_step + step_size / 2.0, approximations_tmp);
		func_buf = func_result.get_buffer();
		for (size_t j = 0; j < init_conditions_size; j++) {
			k[2][j] = step_size * func_buf[j];
			// set up for k[3]
			approx_tmp_buf[j] = approx_buf[j] + k[2][j];
		}

		func_result
			= system_of_eqs(cur_step + step_size, approximations_tmp);
		func_buf = func_result.get_buffer();
		for (size_t j = 0; j < init_conditions_size; j++) {
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


double derivative_at_index(const TensorView& data,
	int target_index)
{
	const auto& data_buffer = data.view_buffer();
	long int max_size = data_buffer.size();
	double est_deriv;


	if (data.get_shape().size() > 1 || data.get_stride()[0] != 1) {
		throw std::runtime_error(
			"Squeeze the tensor such that it is 1 dimension and dense"
		);
	}

	// boundary checks
	if (target_index < 0 || target_index >= max_size) {
		throw std::runtime_error(
			"Each point needs at least one neighbor side");
	}


	// do five-point midpoint
	else if (target_index > 1 && target_index + 2 < max_size) {
		int i = target_index;
		est_deriv = (1.0 / 12.0)
			* (
				data_buffer[i - 2] - 8 * data_buffer[i - 1] + 8 * data_buffer[i + 1]
				- data_buffer[i + 2]);
	}
	// do three-point midpoint
	else if (target_index >= 1 && target_index + 2 <= max_size) {
		est_deriv = (1.0 / 2.0)
			* (-data_buffer[target_index - 1] + data_buffer[target_index + 1]);
	}
	// do a forward guess to keep bounds
	else if (target_index == 0) {
		est_deriv = data_buffer[1] - data_buffer[0];
	}
	// do a backward guess to keep bounds
	else {
		est_deriv = data_buffer[max_size - 1] - data_buffer[max_size - 2];
	}

	return est_deriv;
}


} // end namespace
