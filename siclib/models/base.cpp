#include "base.hpp"
#include <exception>
#include <iostream>
namespace sic
{

void LinearModel::fit_model(
	const TensorView& x_vals,
	const TensorView& y_vals_base
)
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

	TensorView x_vals_transpose = x_vals.transpose();
	m_coefficients =
		x_vals_transpose.matmul(x_vals).transpose()
		.matinv()
		.matmul(x_vals_transpose).matmul(y_vals);

}

TensorView LinearModel::predict(
	const TensorView& x_vals_base
) const
{
	TensorView x_vals = x_vals_base;
	if (x_vals.is_vector()) {
		x_vals = x_vals.unsqueeze(0);
	}

	return x_vals.matmul(m_coefficients);
}

} //
