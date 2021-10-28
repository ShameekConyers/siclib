#pragma once

#include "../linalg.hpp"

#include <random>


namespace sic
{

static std::mt19937 siclib_rng{ 0 };

TensorView find_moment(TensorView& target, size_t dim, size_t moment,
	bool central = false, bool standardized = false);


TensorView find_mean(TensorView& target, size_t dim);
TensorView find_variance(TensorView& target, size_t dim);
TensorView find_stddev(TensorView& target, size_t dim);
TensorView find_skew(TensorView& target, size_t dim);
TensorView find_kurtosis(TensorView& target, size_t dim);

TensorView rand_normal_tensor(
	double mean,
	double stddev,
	std::vector<size_t> shape);

TensorView rand_uniform_tensor(
	double left,
	double right,
	std::vector<size_t> shape);
}
