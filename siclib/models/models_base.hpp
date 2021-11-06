#pragma once
#include "../linalg.hpp"

namespace sic
{

TensorView prepare_fit(const TensorView& x_vals, const TensorView& y_vals_base);
TensorView prepare_prediction(const TensorView& x_input_base);


struct LinearModel {
	enum FitProcedure {
		OrdinaryLeastSquares = 0,
	};

	LinearModel()
	{

	};
	void fit_model(
		const TensorView& x_vals,
		const TensorView& y_vals_base,
		ushort fit_procedure = 0
	);

	TensorView predict(
		const TensorView& x_input_base
	) const;

	TensorView m_coefficients;
};


struct KNearestNeighbors {
	enum Metric {
		EuclideanDistance = 0,
	};

	KNearestNeighbors()
	{

	};

	void fit_model(
		const TensorView& x_vals,
		const TensorView& y_vals_base,
		size_t num_neighbors,
		ushort metric = 0
	);

	TensorView predict(
		const TensorView& x_input_base
	);

	TensorView calculate_distances(
		const TensorView& x_input
	);

	size_t m_num_neighbors;
	TensorView m_x_vals;
	TensorView m_y_vals;
	ushort m_metric;
};


} // end namespace
