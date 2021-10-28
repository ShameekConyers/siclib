#pragma once

#include <functional>
#include <optional>
#include "../linalg.hpp"

namespace sic
{

std::optional<double> equation_solution(
	const std::function<double(double)>& input_func,
	double target_val
);


// uses composite simpson's rule
double integral_index_interval(
	const TensorView& input,
	size_t start_index,
	size_t end_index
);


TensorView initial_value_problem(
	std::function <TensorView(
		double, TensorView)>& system_of_eqs,
	const TensorView& initial_conditions,
	double target_value,
	double initial_value
);

// uses five point midpoint
double derivative_at_index(
	const TensorView& data,
	int target_index
);


} // end namespace
