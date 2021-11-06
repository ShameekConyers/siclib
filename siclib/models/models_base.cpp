#include "models_base.hpp"
#include <exception>
#include <iostream>
#include <utility>
#include <unordered_map>

namespace sic
{

TensorView prepare_fit(const TensorView& x_vals, const TensorView& y_vals_base)
{
	TensorView y_vals = y_vals_base;
	if (!x_vals.is_matrix()) {
		throw std::runtime_error("x_vals should be a matrix");
	}
	if (y_vals.is_vector()) {
		y_vals = y_vals.transpose();
	}
	if (!y_vals.is_matrix()) {
		throw std::runtime_error("y_vals should be a vector or 1 dim matrix");
	}
	if (x_vals.get_shape()[0] != y_vals.get_shape()[0]) {
		throw std::runtime_error("y_vals should mat dim of x_vals");
	}
	return y_vals;
}

TensorView prepare_prediction(const TensorView& x_input_base)
{
	TensorView x_input = x_input_base;
	if (x_input.is_vector()) {
		x_input = x_input.unsqueeze(0);
	}
	if (!x_input.is_matrix() || x_input.get_shape()[0] != 1) {
		throw std::runtime_error("x_input not proper shape");
	}
	return x_input;
}


void LinearModel::fit_model(
	const TensorView& x_vals,
	const TensorView& y_vals_base,
	ushort fit_procedure
)
{
	TensorView y_vals = prepare_fit(x_vals, y_vals_base);

	switch (fit_procedure) {
		case OrdinaryLeastSquares:
		{
			TensorView x_vals_transpose = x_vals.transpose();
			m_coefficients =
				x_vals_transpose.matmul(x_vals).transpose()
				.matinv()
				.matmul(x_vals_transpose).matmul(y_vals);
			break;
		}
		default:
		{
			throw std::runtime_error("Not a valid fit procedure");
			break;
		}
	}


}

TensorView LinearModel::predict(
	const TensorView& x_input_base
) const
{
	TensorView x_input = prepare_prediction(x_input_base);

	return x_input.matmul(m_coefficients);
}


void KNearestNeighbors::fit_model(
	const TensorView& x_vals,
	const TensorView& y_vals_base,
	size_t num_neighbors,
	ushort metric
)
{
	m_y_vals = prepare_fit(x_vals, y_vals_base);
	m_x_vals = x_vals;
	m_metric = metric;
	m_num_neighbors = std::min(num_neighbors, m_y_vals.get_shape()[0]);

	switch (metric) {
		case EuclideanDistance:
		{
			// metric is used during fit currently...
			break;
		}
		default:
		{
			throw std::runtime_error("Not a valid metric");
			break;
		}
	}
}

TensorView KNearestNeighbors::predict(
	const TensorView& x_input_base
)
{
	TensorView x_input = prepare_prediction(x_input_base);

	TensorView dist_vec = calculate_distances(x_input);

	std::vector<std::pair<double, ssize_t>> dist_index_vec;
	const std::vector<double>& buffer = dist_vec.get_buffer();
	std::vector < std::pair< TensorView, size_t >> counter;


	for (ssize_t i = 0; i < buffer.size(); i++) {

		dist_index_vec.push_back({ buffer[i], i });
	}
	std::sort(dist_index_vec.begin(), dist_index_vec.end());


	for (ssize_t i = 0; i < m_num_neighbors; i++) {
		ssize_t target_index = dist_index_vec[i].second;

		TensorView v = m_y_vals.slice_view({ target_index, -1 });

		bool not_found = true;
		for (size_t i = 0; i < counter.size(); i++) {
			if (counter[i].first == v) {
				counter[i].second += 1;
				not_found = false;
				break;
			}
		}
		if (not_found) {
			counter.push_back({ v, 1 });
		}
	}

	size_t best_val = 0;
	TensorView best_y_val;
	for (auto& t : counter) {
		if (t.second > best_val) {
			best_y_val = t.first;
			best_val = t.second;
		}
	}

	return best_y_val;
}

TensorView KNearestNeighbors::calculate_distances(
	const TensorView& x_input
)
{
	TensorView result;

	switch (m_metric) {
		case EuclideanDistance:
		{

			auto square = [](double x)
			{
				return pow(x, 2);
			};

			auto square_root = [](double x)
			{
				return sqrt(x);
			};

			result = m_x_vals
				.binary_element_wise_op(x_input, std::minus())
				.unary_op(square)
				.fold_op(std::plus(), 0, 1)
				.unary_op(square_root);
			break;
		}
		default:
		{
			throw std::runtime_error("Not a valid metric");
			break;
		}
	}

	return result;
}

} //
