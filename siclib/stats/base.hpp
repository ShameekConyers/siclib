#pragma once

#include "../linalg.hpp"

namespace sic
{

TensorView find_moment(TensorView& target, size_t dim, size_t moment,
	bool central = false, bool standardized = false);


TensorView find_mean(TensorView& target, size_t dim);
TensorView find_variance(TensorView& target, size_t dim);
TensorView find_stddev(TensorView& target, size_t dim);
TensorView find_skew(TensorView& target, size_t dim);
TensorView find_kurtosis(TensorView& target, size_t dim);

TensorView find_covariance(TensorView& target, size_t dim);


TensorView rand_normal(std::vector<size_t> number, size_t mean);
}
