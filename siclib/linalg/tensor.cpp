#include "tensor.hpp"
#include <iostream>

namespace sic
{
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


TensorView::TensorView(
	const TensorView& other_view,
	std::vector<size_t> input_shape,
	std::vector<size_t> input_stride,
	size_t offset)
{

	const std::vector<double>& input_data = other_view.m_storage->m_data;
	// sanity checks
	set_shape_stride(input_data,
		input_shape,
		input_stride,
		offset
	);

	m_storage = other_view.m_storage;
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
{
	size_t target_index = 0;
	size_t shape_size = 0;
	size_t counter = 0;
	if (selection.size() != m_shape.size()) {
		throw std::runtime_error("selection not valid.");
	}
	for (size_t i = 0; i < selection.size(); i++) {
		if (selection[i] == 0) {
			continue;
		}
		else if (selection[i] < m_shape[counter]) {
			target_index += selection[i] * m_stride[counter];
			counter++;
		}
		else {
			throw std::runtime_error("selection not valid.");
		}
	}
	return m_storage->m_data[target_index];
}

void TensorView::set_val(const std::vector<size_t>& selection, double val)
{
	size_t target_index = 0;
	size_t shape_size = 0;
	size_t counter = 0;
	if (selection.size() != m_shape.size()) {
		throw std::runtime_error("selection not valid.");
	}
	for (size_t i = 0; i < selection.size(); i++) {
		if (selection[i] == 0) {
			continue;
		}
		else if (selection[i] < m_shape[counter]) {
			target_index += selection[i] * m_stride[counter];
			counter++;
		}
		else {
			throw std::runtime_error("selection not valid.");
		}
	}
	m_storage->m_data[target_index] = val;
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
	std::vector<size_t> this_shape = m_shape;
	std::vector<size_t> other_shape = other.m_shape;
	size_t res_size = std::max(this_shape.size(), other_shape.size());
	std::vector<double> res_data;

	std::vector<size_t> res_shape = do_broadcast(other);
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
					cur_other[c_idx] = std::min(cur_res[c_idx], this_shape[c_idx] - 1);
					cur_this[c_idx] = std::min(cur_res[c_idx], other_shape[c_idx] - 1);
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


std::vector<size_t> TensorView::do_broadcast(TensorView& other)
{
	m_saved_shape = m_shape;
	m_saved_stride = m_stride;


	std::vector<size_t> other_shape = std::move(other.m_shape);
	size_t res_size = std::max(m_shape.size(), other_shape.size());
	std::vector<size_t> res_shape(res_size);

	if (m_shape.size() != other_shape.size()) {
		std::vector<size_t> tmp_shape;
		std::vector<size_t> tmp_stride;
		if (m_shape.size() < other_shape.size()) {
			tmp_shape.resize(res_size - m_shape.size(), 1);
			tmp_shape.resize(res_size - m_shape.size(), m_stride[0]);
			tmp_stride.insert(
				tmp_shape.end(), other.m_stride.begin(), other.m_stride.end());
			m_shape = std::move(tmp_shape);
		}
		else {
			tmp_shape.resize(res_size - other_shape.size(), 1);
			tmp_stride.resize(res_size - m_shape.size(), other.m_stride[0]);
			tmp_shape.insert(
				tmp_shape.end(), other_shape.begin(), other_shape.end());
			tmp_stride.insert(
				tmp_shape.end(), m_stride.begin(), m_stride.end());
			other.m_shape = std::move(tmp_shape);
		}
	}

	for (size_t i = 0; i < res_size; i++) {
		size_t l = m_shape[i];
		size_t r = other_shape[i];

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


	input_stride.resize(input_shape.size());

	size_t shape_prod = 1;
	for (size_t i = 0; i < input_stride.size(); i++) {

		shape_prod *= input_shape[i] * input_stride[i];
		size_t prev = 1;
		if (i == 0) {
			prev = input_stride[i - 1] * input_shape[i - 1];
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


}
