
#include "TensorView.hpp"
#include <iostream>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <OpenBlas/cblas.h>
#endif

namespace sic
{
/**
 * @brief
 *
 * @param selection where -1 means to "get all"
 * @return TensorView
 */
TensorView TensorView::slice_view(
	const std::vector<ssize_t>& selection) const
{
	size_t target_index = 0;
	size_t shape_size = 0;
	std::vector<size_t> new_shape = m_shape;
	bool sliced_once = false;

	if (selection.size() != m_shape.size()) {
		throw std::runtime_error("slice not valid. e1");
	}
	for (size_t i = 0; i < selection.size(); i++) {

		// shape buisiness
		if (selection[i] == -1) {
			new_shape[i] = m_shape[i];
			sliced_once = true;
		}
		else {
			new_shape[i] = 1;
		}

		// index buisiness
		if (selection[i] <= 0) {
			continue;
		}
		else if (selection[i] < m_shape[i]) {
			target_index += selection[i] * m_stride[i];
		}
		else {
			throw std::runtime_error("slice not valid. e2");
		}
	}
	if (!sliced_once) {
		throw std::runtime_error("use -1 to select a dimension to slice");
	}

	size_t new_offset = m_offset + target_index;

	TensorView result = { *this, new_shape, m_stride, new_offset };
	return result;
}

double TensorView::get_val(const std::vector<size_t>& selection)
const
{

	size_t target_index = 0;
	size_t shape_size = 0;
	if (selection.size() != m_shape.size()) {
		throw std::runtime_error("selection not valid. e1");
	}
	for (size_t i = 0; i < selection.size(); i++) {
		if (selection[i] == 0) {
			continue;
		}
		else if (selection[i] < m_shape[i]) {
			target_index += selection[i] * m_stride[i];
		}
		else {
			throw std::runtime_error("selection not valid. e2");
		}
	}
	return (*m_storage)[m_offset + target_index];
}

void TensorView::set_val(const std::vector<size_t>& selection, double val)
{


	size_t target_index = 0;
	size_t shape_size = 0;
	if (selection.size() != m_shape.size()) {
		throw std::runtime_error("selection not valid.");
	}
	for (size_t i = 0; i < selection.size(); i++) {
		if (selection[i] == 0) {
			continue;
		}
		else if (selection[i] < m_shape[i]) {
			target_index += selection[i] * m_stride[i];
		}
		else {

			throw std::runtime_error("selection not valid. e2");
		}
	}
	(*m_storage)[m_offset + target_index] = val;
}

TensorView TensorView::operator- (TensorView& other)
{
	return binary_element_wise_op(other, std::minus());
}
TensorView TensorView::operator+ (TensorView& other)
{
	return binary_element_wise_op(other, std::plus());
}



TensorView TensorView::matmul(const TensorView& other) const
{
	// checks to see if operation is valid
	std::vector<size_t> other_shape = other.get_shape();

	//  check  for compatible shape
	if (!is_matrix() || !other.is_matrix()
		|| m_shape[1] != other_shape[0]) {
		throw std::runtime_error("incompatible shapes");
	}

	size_t l = m_shape[0];
	size_t r = other_shape[1];
	size_t compat = m_shape[1];

	std::vector<size_t> result_shape = { l, r };
	TensorView result = generate_tensor(result_shape, 0);

	/* M_ij = bin_op ith Left and jth Right (transposed) and accum */
	// for (size_t i = 0; i < l; i++) {
	// 	for (size_t j = 0; j < r; j++) {

	// 		TensorView ith_row = slice_view({ (ssize_t)i, -1 }).squeeze();

	// 		TensorView jth_col = other.slice_view({ -1, (ssize_t)j }).squeeze();

	// 		double dotprod = ith_row.dotprod(jth_col).get_item();

	// 		result.set_val({ i, j }, dotprod);

	// 	}
	// }


	auto lflag = CblasNoTrans;
	auto rflag = CblasNoTrans;
	TensorView lten = *this;
	TensorView rten = other;
	if (!is_aligned()) {
		lten = do_mat_alignment();
	}
	if (!other.is_aligned()) {
		rten = other.do_mat_alignment();
	}
	cblas_dgemm(
		CblasRowMajor,
		lflag,
		rflag,
		(int)l,
		(int)r,
		(int)compat,
		1.0,
		lten.m_storage->data(),
		(int)compat,
		rten.m_storage->data(),
		(int)r,
		1.0,
		const_cast<double*>(result.m_storage->data()),
		(int)r
	);
	return result;
}

TensorView TensorView::matinv() const
{
	if (!is_matrix()) {
		throw std::runtime_error("Not a Matrix");
	}
	if (m_shape[0] != m_shape[1]) {
		throw std::runtime_error("Not a Square Matrix");
	}
	TensorView this_col_major = switch_mat_major_order();

	int mat_dim = m_shape[1];
	int LWORK = mat_dim * mat_dim;

	int INFO = 0;

	std::vector<int> IPIV(mat_dim);
	std::vector<double> WORK(LWORK);

	dgetrf_(
		&mat_dim,
		&mat_dim,
		const_cast<double*>(this_col_major.m_storage->data()),
		&mat_dim,
		const_cast<int*>(IPIV.data()),
		&INFO)
		;

	dgetri_(
		&mat_dim,
		const_cast<double*>(this_col_major.m_storage->data()),
		&mat_dim,
		const_cast<int*>(IPIV.data()),
		const_cast<double*>(WORK.data()),
		&LWORK,
		&INFO
	);

	return this_col_major.switch_mat_major_order();

}

TensorView TensorView::dotprod(const TensorView& other) const
{
	assert(is_vector() && other.is_vector());

	TensorView element_prods = binary_element_wise_op(
		other, std::multiplies());

	assert(element_prods.m_shape.size() == 1);

	return element_prods.fold_op(std::plus(), 0, 0);
}


std::tuple<std::vector<size_t>, TensorView, TensorView>
TensorView::do_broadcast(
	const TensorView& other) const
{
	std::vector<size_t> brd_this_shape = m_shape;
	std::vector<size_t> brd_this_stride = m_stride;

	std::vector<size_t> brd_other_shape = other.m_shape;
	std::vector<size_t> brd_other_stride = other.m_stride;
	size_t other_dim = other.m_shape.size();
	size_t this_dim = m_shape.size();

	size_t res_size = std::max(m_shape.size(), other.m_shape.size());
	std::vector<size_t> res_shape(res_size);

	if (this_dim != other_dim) {
		if (this_dim < other_dim) {
			brd_this_shape.insert(
				brd_this_shape.begin(),
				1,
				other_dim - this_dim);
			brd_this_stride.insert(
				brd_this_stride.begin(),
				m_stride[0] * m_shape[0],
				other_dim - this_dim);
		}
		else {
			brd_other_shape.insert(
				brd_other_shape.begin(),
				1,
				this_dim - other_dim);
			brd_other_stride.insert(
				brd_other_stride.begin(),
				other.m_stride[0] * other.m_shape[0],
				this_dim - other_dim);
		}
	}
	TensorView brd_this{
		 *this, brd_this_shape, brd_this_stride, m_offset };
	TensorView brd_other{
		other, brd_other_shape, brd_other_stride, other.m_offset };

	for (size_t i = 0; i < res_size; i++) {
		size_t l = m_shape[i];
		size_t r = other.m_shape[i];

		if (l != r && (l != 1 && r != 1)) {
			throw std::runtime_error("incompatible shapes");
		}
		res_shape[i] = std::max(r, l);
	}

	return { res_shape, brd_this, brd_other };
}

/**
 * @brief
 *
 * @param target_dim if -1 squeezes all dims
 * @return TensorView
 */
TensorView TensorView::squeeze(ssize_t target_dim) const
{
	if (target_dim != -1) {
		if (target_dim >= m_shape.size()) {
			throw std::runtime_error("dimension doesn't exist");
		}
		else if (m_shape[target_dim] != 1) {
			std::runtime_error("non-zero dimension");
		}

		TensorView result = *this;
		result.m_shape.erase(result.m_shape.begin() + target_dim);
		result.m_stride.erase(result.m_stride.begin() + target_dim);
		return result;
	}
	else {

		std::vector<size_t> new_shape;
		std::vector<size_t> new_stride;

		for (size_t i = 0; i < m_shape.size(); i++) {
			if (m_shape[i] != 1) {
				new_shape.push_back(m_shape[i]);
				new_stride.push_back(m_stride[i]);
			}
		}
		if (new_shape.size() == 0) {
			new_shape.push_back(1);
			new_stride.push_back(1);
		}

		TensorView result = *this;
		result.m_shape = new_shape;
		result.m_stride = new_stride;
		return result;
	}
}

TensorView TensorView::unsqueeze(size_t target_dim) const
{
	// TODO check this
	size_t target_dim_stride = 1;
	if (target_dim > m_shape.size()) {
		throw std::runtime_error("cannot add a dimension here");
	}

	if (target_dim + 1 == m_shape.size()) {
		target_dim_stride = m_stride[target_dim];
	}
	else if (target_dim + 1 < m_shape.size()) {
		target_dim_stride = m_stride[target_dim + 1] * m_shape[target_dim + 1];
	}
	else if (target_dim == m_shape.size()) {
		target_dim_stride = m_stride[target_dim - 1];
	}
	else {
		throw std::logic_error("...");
	}
	std::vector<size_t> new_shape = m_shape;
	new_shape.insert(new_shape.begin() + target_dim, 1);
	std::vector<size_t> new_stride = m_stride;
	new_stride.insert(new_stride.begin() + target_dim, target_dim_stride);


	return { *this, new_shape, new_stride, m_offset };
}
void TensorView::set_shape_stride(const std::vector<double>& input_data,
	std::vector<size_t>& input_shape,
	std::vector<size_t>& input_stride,
	size_t offset)
{


	if (offset > input_data.size()) {
		throw std::runtime_error("invalid offset");
	}
	size_t ref_size = input_data.size() - offset;

	if (input_shape.size() == 0) {
		input_shape = { input_data.size() };
	}

	bool gen_stride = false;
	if (input_stride.size() == 0) {
		input_stride.resize(input_shape.size(), 1);
		gen_stride = true;
	}
	else if (input_stride.size() == input_shape.size());
	else {
		throw std::runtime_error("invalid stride e1");
	}


	size_t shape_prod = 1;
	size_t largest_stride = 0;
	size_t largest_stride_index = 0;
	for (ssize_t i = input_stride.size() - 1; i >= 0; i--) {

		if (input_stride[i] >= largest_stride) {
			largest_stride = input_stride[i];
			largest_stride_index = i;
		}
		if (gen_stride) {
			if (i == input_stride.size() - 1) {
				input_stride[i] = 1;
			}
			else {
				input_stride[i] = input_shape[i + 1];
			}
		}
	}
	shape_prod = largest_stride * input_shape[largest_stride_index];
	if (shape_prod > ref_size) {
		throw std::runtime_error("invalid shape");
	}

	m_stride = input_stride;
	m_shape = input_shape;
	m_offset = offset;
}

TensorView TensorView::transpose(ssize_t dim_1, ssize_t dim_2) const
{
	if (dim_1 != -1 && dim_2 != -1) {
		if (dim_1 >= m_shape.size() || dim_2 >= m_shape.size()) {
			throw std::runtime_error("invalid dimensions: transpose");
		}

		std::vector<size_t> result_shape = m_shape;
		std::vector<size_t> result_stride = m_stride;

		std::swap(result_shape[dim_1], result_shape[dim_2]);
		std::swap(result_stride[dim_1], result_stride[dim_2]);

		TensorView result{ *this };
		result.m_shape = result_shape;
		result.m_stride = result_stride;

		return result;
	}
	else if (m_shape.size() == 1) {
		return unsqueeze(1);
	}
	else {
		std::vector<size_t> result_shape = m_shape;
		std::vector<size_t> result_stride = m_stride;
		std::reverse(result_shape.begin(), result_shape.end());
		std::reverse(result_stride.begin(), result_stride.end());
		TensorView result{ *this };
		result.m_shape = result_shape;
		result.m_stride = result_stride;

		return result;
	}
}


TensorView generate_tensor(std::vector<size_t> shape, double inital_value)
{
	size_t cumul_prod = 1;
	for (size_t i = 0; i < shape.size(); i++) {
		cumul_prod *= shape[i];
	}

	std::vector<double> data;
	data.resize(cumul_prod, inital_value);
	TensorView result{ data, shape };
	return result;
}


std::vector<size_t> TensorView::get_shape() const
{
	return m_shape;
}

std::vector<size_t> TensorView::get_stride() const
{
	return m_stride;
}

size_t TensorView::get_offset() const
{
	return m_offset;
}

double TensorView::get_item() const
{
	size_t cumul_prod = 1;
	for (auto i : m_shape) {
		cumul_prod *= i;
	}
	if (cumul_prod != 1) {
		throw std::runtime_error("not a singleton tensor");
	}
	else {
		return get_val(std::vector<size_t>(m_shape.size()));
	}
}

bool TensorView::is_matrix() const
{
	if (m_shape.size() != 2) {
		return false;
	}
	return true;
}

bool TensorView::is_vector() const
{
	if (m_shape.size() != 1) {
		return false;
	}
	return true;
}

TensorView TensorView::do_mat_alignment() const
{
	std::vector<double> result;

	for (size_t i = 0; i < m_shape[0]; i++) {
		for (size_t j = 0; j < m_shape[1]; j++) {
			result.push_back(get_val({ i, j }));
		}
	}

	return { result, m_shape };
}

bool TensorView::is_aligned() const
{

	for (size_t i = 1; i < m_stride.size(); i++) {
		if (m_stride[i] > m_stride[i - 1]) {
			return false;
		}
	}
	return true;
}

TensorView TensorView::switch_mat_major_order() const
{
	if (!is_matrix()) {
		throw std::runtime_error("Not a Matrix");
	}

	TensorView result = transpose().do_mat_alignment().transpose();
	return result;
}
} //
