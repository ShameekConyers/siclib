#include "descriptive.hpp"
#include <exception>
#include <functional>
#include <iostream>

namespace sic
{

TensorView find_moment(
	TensorView& target, size_t dim, size_t moment, bool central, bool standardized)
{
	if (moment < 1) {
		throw std::runtime_error("Moment needs to be at least 1");
	}
	if (dim > target.get_shape().size()) {
		throw std::runtime_error("dim doesn't exist");
	}

	TensorView mean_tensor = target;
	double num_in_dim = target.get_shape()[dim];
	if (central || standardized) {
		mean_tensor = find_moment(target, dim, 1);
	}
	else {
		std::vector<size_t> desired_dim = target.get_shape();
		mean_tensor = generate_tensor(desired_dim, 0);
	}

	double moment_num = moment;
	std::function<double(double, double)> proc = [moment_num, num_in_dim](
		double lhs, double rhs)
	{
		auto res = lhs - rhs;

		return pow(res, moment_num) / num_in_dim;
	};

	TensorView result = target.binary_element_wise_op(mean_tensor, proc);

	if (standardized) {
		TensorView var_tensor = find_moment(target, dim, 2, true);

		std::function<double(double, double)> standard_proc = [](
			double lhs, double rhs)
		{
			return lhs / (sqrt(rhs));
		};

		result = result.binary_element_wise_op(var_tensor, standard_proc);
	}

	// Now we accumulate over target dim
	std::function<double(double, double)> accumulate = [](double rhs, double lhs)
	{
		return rhs + lhs;
	};

	result = result.fold_op(accumulate, 0, dim);

	return result;
}

TensorView find_mean(TensorView& target, size_t dim)
{
	return find_moment(target, dim, 1);
}

TensorView find_variance(TensorView& target, size_t dim)
{
	return find_moment(target, dim, 2, true);
}

TensorView find_stddev(TensorView& target, size_t dim)
{
	TensorView var_tensor = find_variance(target, dim);
	std::function<double(double)> proc = [](double lhs)
	{
		return powl(lhs, 0.5);
	};

	return var_tensor.unary_op(proc);
}

TensorView find_skew(TensorView& target, size_t dim)
{
	return find_moment(target, dim, 3, true, true);
}

TensorView find_kurtosis(TensorView& target, size_t dim)
{
	return find_moment(target, dim, 4, true, true);
}


TensorView rand_normal_tensor(double mean,
	double stddev,
	std::vector<size_t> shape)
{
	std::normal_distribution<double> distr{ mean, stddev };

	std::vector<double> result_data;
	size_t cumul_prod = 1;
	for (auto item : shape) {
		cumul_prod *= item;
	}

	result_data.reserve(cumul_prod);
	for (size_t i = 0; i < cumul_prod; i++) {
		result_data.push_back(distr(siclib_rng));
		std::cerr << i << "...";
	}

	return TensorView{ result_data, shape };
}


TensorView rand_uniform_tensor(double left,
	double right,
	std::vector<size_t> shape)
{

	std::uniform_real_distribution<double> distr{ left, right };

	std::vector<double> result_data;
	size_t cumul_prod = 1;
	for (auto item : shape) {
		cumul_prod *= item;
	}

	result_data.reserve(cumul_prod);
	for (size_t i = 0; i < cumul_prod; i++) {
		result_data.push_back(distr(siclib_rng));
	}
	// std::cerr << cumul_prod;
	// std::cerr << "[" << shape[0] << ", " << shape[1] << "]" << "\n";

	return TensorView{ result_data, shape };
}


}
