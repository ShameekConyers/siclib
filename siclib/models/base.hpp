#pragma once
#include "../linalg.hpp"

namespace sic
{


class SupervisedModel {
	void fit_model(const TensorView& x_vals, const TensorView& y_vals);

};

class LinearModel : public SupervisedModel {
	void fit_model(
		const TensorView& x_vals,
		const TensorView& y_vals
	);

	TensorView predict(
		const TensorView& x_vals
	);

	TensorView m_coefficients;
};


class KNearestNeighbors : public SupervisedModel {

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
