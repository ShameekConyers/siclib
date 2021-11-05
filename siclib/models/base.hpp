#pragma once
#include "../linalg.hpp"

namespace sic
{



struct LinearModel {
	LinearModel()
	{

	};
	void fit_model(
		const TensorView& x_vals,
		const TensorView& y_vals
	);

	TensorView predict(
		const TensorView& x_vals
	) const;

	TensorView m_coefficients;
};


class KNearestNeighbors {

	void fit_model(
		const TensorView& x_vals,
		const TensorView& y_vals,
		size_t num_neighbors
	);

	void predict(
		const TensorView& x_vals
	);

	size_t m_num_neighbors;
};


} // end namespace
