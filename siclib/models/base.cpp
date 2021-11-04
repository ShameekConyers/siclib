#include "base.hpp"
#include <exception>
namespace sic
{

void LinearModel::fit_model(
	const TensorView& x_vals,
	const TensorView& y_vals_base
)
{
	TensorView y_vals = y_vals_base.squeeze();
	if (!x_vals.is_matrix()) {
		throw std::runtime_error("x_vals should be a matrix");
	}
	if (!y_vals.is_vector()) {
		throw std::runtime_error("y_vals should be a vector or 1 dim matrix");
	}
	if (x_vals.get_shape()[0] != y_vals.get_shape()[0]) {
		throw std::runtime_error("y_vals should mat dim of x_vals");
	}

	// TensorView coefficents;
}

} //
