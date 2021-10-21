#include "Tensor.hpp"
#include <numeric>
#include <exception>
#include <iostream>

namespace sic
{
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
	m_storage->m_data = input_data;
}

TensorView::TensorView(
	TensorView& other_view,
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
	m_storage->m_data = input_data;
}

TensorView TensorView::deep_copy()
{
	return TensorView{ m_storage->m_data, m_shape, m_stride };
}

double TensorView::get_val(const std::vector<size_t>& selection)
{
	size_t target_index = 0;
	size_t shape_size = 0;
	size_t counter = 0;

	for (size_t i = 0; i < selection.size(); i++) {
		if (selection[i] == 0) {
			continue;
		}
		else if (selection[i] < m_shape.at(counter)) {
			target_index += selection[i] * m_traverse_offset[counter];
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
	if (selection.size() < m_shape.size()) {
		throw std::runtime_error("selection not valid.");
	}
	for (size_t i = 0; i < selection.size(); i++) {
		if (selection[i] == 0) {
			continue;
		}
		else if (selection[i] < m_shape.at(counter)) {
			target_index += selection[i] * m_traverse_offset[counter];
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
	// TODO
	int c_idx = 0;
	int counter = 0;
	size_t cumulative_prod = 1;
	std::vector<size_t> this_shape = m_shape;
	std::vector<size_t> other_shape = other.m_shape;
	size_t res_size = std::max(this_shape.size(), other_shape.size());
	std::vector<size_t> res_shape(res_size);
	std::vector<double> res_data;


	if (this_shape.size() != other_shape.size()) {
		std::vector<size_t> tmp;
		if (this_shape.size() < other_shape.size()) {
			tmp.resize(res_size - this_shape.size(), 1);
			tmp.insert(tmp.end(), this_shape.begin(), this_shape.end());
			this_shape = std::move(tmp);
		}
		else {
			tmp.resize(res_size - other_shape.size(), 1);
			tmp.insert(tmp.end(), other_shape.begin(), other_shape.end());
			other_shape = std::move(tmp);
		}
	}

	for (size_t i = 0; i < res_size; i++) {
		size_t l = this_shape[i];
		size_t r = other_shape[i];

		if (l != r && (l != 1 && r != 1)) {
			throw std::runtime_error("incompatible shapes");
		}
		res_shape[i] = std::max(r, l);
		cumulative_prod *= res_shape[i];
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
	return res_tensor;
}

void TensorView::update_traverse()
{

	m_traverse_offset.resize(m_shape.size());

	for (ssize_t i = m_traverse_offset.size() - 1; i >= 0; i--) {

		if (i != m_traverse_offset.size() - 1) {
			m_traverse_offset[i] = m_traverse_offset[i + 1] * m_stride[i];
		}
		else {
			m_traverse_offset[i] = m_stride[i] * 1;
		}
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
		throw std::runtime_error("invalid stride");
	}
	input_stride.resize(input_shape.size(), 1);
	size_t shape_prod = 1;
	for (size_t i = 0; i < input_stride.size(); i++) {
		if (input_stride[i] > input_shape[i]) {
			throw std::runtime_error("invalid stride");
		}
		if (input_shape[i] % input_stride[i] != 0) {
			throw std::runtime_error("invalid stride");
		}
		shape_prod = input_shape[i] * input_stride[i];
	}

	if (shape_prod > ref_size) {
		throw std::runtime_error("invalid shape");
	}
	if (shape_prod == 0) {
		throw std::runtime_error("invalid shape");
	}

	m_stride = input_stride;
	m_shape = input_shape;
	m_offset = offset;
	update_traverse();
}


}
