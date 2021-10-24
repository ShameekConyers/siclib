#include "Tensor.hpp"
#include <iostream>

namespace sic
{

template <typename T>
std::ostream& operator<<(std::ostream& output, std::vector<T> const& values)
{
	output << "[";
	for (size_t i = 0; i < values.size(); i++) {
		output << values[i];
		if (i != values.size() - 1) output << ", ";
	}
	output << "]\n";
	return output;
}

std::ostream& operator<<(std::ostream& output, const TensorView& view)
{
	output << "Tensor: \n";
	int c_idx = 0;
	int counter = 1;

	std::vector<size_t> this_shape = view.m_shape;


	std::vector<size_t> cur_this(this_shape.size());
	size_t num_spaces = 0;

	bool flag = true;
	ssize_t lst_sig = this_shape.size() - 1;
	for (size_t i = 0; i < this_shape.size(); i++) output << "[";
	while (flag) {

		//! Insert print here
		output << view.get_val(cur_this);

		cur_this[lst_sig]++;


		if (cur_this[lst_sig] == this_shape[lst_sig]) {

			c_idx = lst_sig;

			while (flag) {
				if (c_idx == -1) {
					flag = false;
					output << "]\n";
					break;
				}
				// output << "*";
				if (cur_this[c_idx] == this_shape[c_idx]) {
					cur_this[c_idx] = 0;
					c_idx--;

				}
				else {

					cur_this[c_idx] += 1;
					if (cur_this[c_idx] < this_shape[c_idx]) {
						output << "]\n";
						num_spaces = this_shape.size() - counter;
						for (size_t i = 0; i < num_spaces; i++) output << " ";
						for (size_t i = 0; i < counter; i++) output << "[";

						counter = 1;
						break;
					}
					else {
						counter++;
						output << "]";

						continue;
					}
				}
			}
		}
		else {
			output << ", ";
		}
	}
	output << "Tensor Shape: ";
	output << view.m_shape << "\n";
	return output;
}

TensorView::TensorView(pybind11::array_t<double> numpy_array)
{
	pybind11::buffer_info np_arr_info = numpy_array.request();
	double* data_begin = (double*)np_arr_info.ptr;

	// numpy arrays strides are in bytes rather than type offset
	// Hence we have to convert them to type offset for interop w/ interators

	//find highest stride and corresponding shape
	ssize_t highest_stride = -1;
	ssize_t highest_index = -1;
	size_t dbl_size = sizeof(double);
	std::vector<size_t> tmp_shape;
	std::vector<size_t> tmp_stride;
	std::vector<std::pair<size_t, size_t>> stride_index_vec;

	for (size_t i = 0; i < np_arr_info.strides.size(); i++) {
		if (np_arr_info.strides[i] > highest_stride) {
			highest_stride = np_arr_info.strides[i];
			highest_index = i;
		}
		tmp_shape.push_back(np_arr_info.shape[i]);
		tmp_stride.push_back(np_arr_info.strides[i] / dbl_size);
	}

	size_t np_arr_bytes;
	np_arr_bytes = tmp_stride[highest_index] * np_arr_info.shape[highest_index];

	std::vector<double> data(data_begin, data_begin + np_arr_bytes);

	m_shape = tmp_shape;
	m_stride = tmp_stride;
	m_offset = 0;
	m_storage = std::make_shared<TensorStorage>(TensorStorage{ data });

}


TensorView::TensorView(
	std::vector<double> input_data,
	std::vector<size_t> input_shape,
	std::vector<size_t> input_stride,
	size_t offset)
{

	// sanity checks
	set_shape_stride(input_data,
		input_shape,
		input_stride,
		offset
	);

	m_storage = std::make_shared<TensorStorage>(TensorStorage{ input_data });
}


std::vector<double>& TensorView::get_buffer()
{
	return m_storage->m_data;
}

TensorView TensorView::deep_copy()
{
	return TensorView{ m_storage->m_data, m_shape, m_stride };
}

pybind11::array_t<double> TensorView::to_numpy()
{
	auto stride_in_bytes = m_stride;
	for (auto i = 0; i < m_stride.size(); i++) {
		stride_in_bytes[i] = stride_in_bytes[i] * sizeof(double);
	}
	pybind11::buffer_info result_info(
		m_storage->m_data.data(),
		sizeof(double),
		pybind11::format_descriptor<double>::format(),
		m_shape.size(),
		m_shape,
		stride_in_bytes
	);
	pybind11::array_t<double> result(result_info);
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
	return m_storage->m_data[target_index];
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
	m_storage->m_data[target_index] = val;
}

TensorView TensorView::operator- (TensorView& other)
{
	return binary_element_wise_op(other, std::minus());
}

TensorView TensorView::unitary_op(
	std::function<double(double)> unitary_op)
{
	int c_idx = 0;
	int counter = 1;

	std::vector<size_t> this_shape = m_shape;


	std::vector<size_t> cur_this(this_shape.size());
	TensorView result_tensor = deep_copy();

	bool flag = true;
	ssize_t lst_sig = this_shape.size() - 1;
	while (flag) {
		//! Insert unitary op here
		double result = unitary_op(get_val(cur_this));
		result_tensor.set_val(cur_this, result);

		cur_this[lst_sig]++;
		if (cur_this[lst_sig] == this_shape[lst_sig]) {
			c_idx = lst_sig;
			while (flag) {
				if (c_idx == -1) {
					flag = false;
					break;
				}
				if (cur_this[c_idx] == this_shape[c_idx]) {
					cur_this[c_idx] = 0;
					c_idx--;
				}
				else {
					cur_this[c_idx] += 1;
					if (cur_this[c_idx] < this_shape[c_idx]) {
						break;
					}
					else {
						continue;
					}
				}
			}
		}
	}
}

TensorView TensorView::operator+ (TensorView& other)
{
	return binary_element_wise_op(other, std::plus());
}

TensorView TensorView::binary_element_wise_op(TensorView& other, std::function<double(double, double)>  binary_op)
{

	int c_idx = 0;
	int counter = 0;
	size_t cumulative_prod = 1;

	std::vector<double> res_data;

	std::vector<size_t> res_shape = do_broadcast(other);
	std::vector<size_t> this_shape = m_shape;
	std::vector<size_t> other_shape = other.m_shape;
	size_t res_size = std::max(this_shape.size(), other_shape.size());

	for (auto item : res_shape) {
		cumulative_prod *= item;
	}
	res_data.resize(cumulative_prod);
	TensorView res_tensor{ res_data, res_shape };
	if (res_data.size() == 0) {
		return res_tensor;
	}


	std::vector<size_t> cur_this(res_shape.size());
	std::vector<size_t> cur_other(res_shape.size());
	std::vector<size_t> cur_res(res_shape.size());

	bool flag = true;
	ssize_t lst_sig = res_size - 1;
	while (flag) {
		//! Insert Binary OP here
		double op_result = binary_op(
			get_val(cur_this), other.get_val(cur_other));
		res_tensor.set_val(cur_res, op_result);

		cur_res[lst_sig]++;
		counter++;
		cur_other[lst_sig] = std::min(this_shape[lst_sig] - 1, cur_res[lst_sig]);
		cur_this[lst_sig] = std::min(other_shape[lst_sig] - 1, cur_res[lst_sig]);
		if (cur_res[lst_sig] == res_shape[lst_sig]) {
			c_idx = lst_sig;
			while (flag) {
				if (c_idx == -1) {
					flag = false;
					break;
				}

				if (cur_res[c_idx] == res_shape[c_idx]) {
					cur_res[c_idx] = 0;
					cur_other[c_idx] = 0;
					cur_this[c_idx] = 0;
					c_idx--;
				}
				else {
					cur_res[c_idx] += 1;
					cur_other[c_idx] = std::min(cur_res[c_idx], other_shape[c_idx] - 1);
					cur_this[c_idx] = std::min(cur_res[c_idx], this_shape[c_idx] - 1);

					if (cur_res[c_idx] < res_shape[c_idx]) {
						break;
					}
					else {
						continue;
					}
				}
			}
		}
	}

	undo_broadcast();
	other.undo_broadcast();
	return res_tensor;
}

TensorView TensorView::fold_op(std::function<double(double, double)> binary_op,
	double inital_value,
	size_t target_dim,
	bool left_op)
{
	int c_idx = 0;
	int counter = 0;

	std::vector<size_t> this_shape = m_shape;
	std::vector<size_t> res_shape = m_shape;
	res_shape[target_dim] = 1;


	size_t res_size = res_shape.size();
	TensorView res_tensor = generate_tensor(res_shape, inital_value);

	std::vector<size_t> cur_this(res_shape.size());
	std::vector<size_t> cur_res(res_shape.size());

	bool flag = true;
	ssize_t lst_sig = res_size - 1;
	while (flag) {
		//! Insert FOLD OP here
		double op_result = binary_op(
			get_val(cur_this), res_tensor.get_val(cur_res));
		res_tensor.set_val(cur_res, op_result);


		cur_this[lst_sig]++;
		counter++;
		cur_res[lst_sig] = std::min(res_shape[lst_sig] - 1, cur_this[lst_sig]);


		if (cur_this[lst_sig] == this_shape[lst_sig]) {
			c_idx = lst_sig;
			while (flag) {
				if (c_idx == -1) {
					flag = false;
					break;
				}

				if (cur_this[c_idx] == this_shape[c_idx]) {
					cur_res[c_idx] = 0;
					cur_this[c_idx] = 0;
					c_idx--;
				}
				else {

					cur_this[c_idx] += 1;
					cur_res[c_idx] = std::min(res_shape[c_idx] - 1, cur_this[c_idx]);

					if (cur_this[c_idx] < this_shape[c_idx]) {
						break;
					}
					else {
						continue;
					}
				}
			}
		}
	}

	return res_tensor;
}


std::vector<size_t> TensorView::do_broadcast(TensorView& other)
{
	m_saved_shape = m_shape;
	m_saved_stride = m_stride;

	other.m_saved_shape = other.m_shape;
	other.m_saved_stride = other.m_stride;
	size_t other_dim = other.m_shape.size();
	size_t this_dim = m_shape.size();

	size_t res_size = std::max(m_shape.size(), other.m_shape.size());
	std::vector<size_t> res_shape(res_size);

	if (this_dim != other_dim) {
		if (this_dim < other_dim) {
			m_shape.insert(
				m_shape.begin(), 1, other_dim - this_dim);
			m_stride.insert(
				m_stride.begin(), m_stride[0], other_dim - this_dim);
		}
		else {
			other.m_shape.insert(
				other.m_shape.begin(), 1, this_dim - other_dim);
			other.m_stride.insert(
				other.m_stride.begin(), other.m_stride[0], this_dim - other_dim);
		}
	}

	for (size_t i = 0; i < res_size; i++) {
		size_t l = m_shape[i];
		size_t r = other.m_shape[i];

		if (l != r && (l != 1 && r != 1)) {
			throw std::runtime_error("incompatible shapes");
		}
		res_shape[i] = std::max(r, l);
	}

	return res_shape;
}

void TensorView::undo_broadcast()
{
	if (m_saved_shape.size() != 0) {
		m_shape = m_saved_shape;
		m_stride = m_saved_stride;

		m_saved_stride.clear();
		m_saved_shape.clear();
	}
}

std::vector<size_t> TensorView::get_shape()
{
	return m_shape;
}

std::vector<size_t> TensorView::get_stride()
{
	return m_stride;
}

size_t TensorView::get_offset()
{
	return m_offset;
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

	if (input_stride.size() > input_shape.size()) {
		throw std::runtime_error("invalid stride e1");
	}

	input_stride.resize(input_shape.size(), 1);

	size_t shape_prod = 1;
	for (ssize_t i = input_stride.size() - 1; i >= 0; i--) {

		shape_prod *= input_shape[i] * input_stride[i];
		size_t prev = 1;
		if (i != input_stride.size() - 1) {
			prev = input_stride[i + 1] * input_shape[i + 1];
		}
		input_stride[i] = input_stride[i]
			* prev;
	}

	if (shape_prod > ref_size) {
		throw std::runtime_error("invalid shape");
	}

	m_stride = input_stride;
	m_shape = input_shape;
	m_offset = offset;
}

TensorView TensorView::transpose()
{
	std::vector<size_t> result_shape = m_shape;
	std::vector<size_t> result_stride = m_stride;
	std::reverse(result_shape.begin(), result_shape.end());
	std::reverse(result_stride.begin(), result_stride.end());
	TensorView result{ *this };
	result.m_shape = result_shape;
	result.m_stride = result_stride;
	return result;
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

} //
