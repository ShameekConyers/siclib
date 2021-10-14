#pragma once

#include <functional>
#include <optional>

std::optional<double> find_equation_solution(
	const std::function<double(double)>& input_func,
	double target_val);
