#pragma once

// not used
#include <vector>
#include <memory>
#include <map>
#include <span>
#include <utility>
#include <functional>
#include <pybind11/numpy.h>


namespace sic
{

using TensorDataType = double;
using TensorStorage = std::vector<TensorDataType>;

class TensorObj {

};

class TensorView : public TensorObj {
public:
	typedef double value_type;

	TensorView(
		pybind11::array_t<double> numpy_array
	);

	TensorView(
		const std::vector<double>& input_data,
		std::vector<size_t> input_shape = {},
		std::vector<size_t> input_stride = {},
		size_t offset = 0
	);

	TensorView(
		const TensorView& other_view
	);

	TensorView(
		const TensorView& other_view,
		std::vector<size_t> input_shape,
		std::vector<size_t> input_stride,
		size_t offset
	);

	TensorView& operator=(const TensorView& other_view);

	// TensorView(TensorView&& other) = default;
	// TensorView& operator=(TensorView&& other) = default;
	const	std::vector<double>& view_buffer() const;
	std::vector<double>& get_buffer();

	TensorView deep_copy() const;
	pybind11::array_t<double> to_numpy();

	TensorView slice_view(const std::vector<ssize_t>& selection) const;
	double get_val(const std::vector<size_t>& selection) const;
	void set_val(const std::vector<size_t>& selection, double val);


	double& operator[] (size_t element); // TODO
	TensorView operator+ (TensorView& other);
	TensorView operator- (TensorView& other);

	template<typename Fn>
	TensorView unary_op(
		const Fn& func
	) const;

	template<typename Fn>
	TensorView binary_element_wise_op(
		const TensorView& other,
		const Fn& func
	) const;

	template<typename Fn>
	TensorView fold_op(
		const Fn& binary_op,
		double inital_value,
		size_t target_dim,
		bool left_op = true
	);

	TensorView transpose(ssize_t dim_1 = -1, ssize_t dim_2 = -1);

	// matrix methods
	TensorView matmul(const TensorView& other) const;
	TensorView mat_inv() const; // TODO

	// see numpy dot
	TensorView dotprod(const TensorView& other) const; // TODO

	TensorView squeeze(ssize_t target_dim = -1) const;
	TensorView unsqueeze(size_t target_dim) const;

	void set_shape_stride(
		const std::vector<double>& reference_data,
		std::vector<size_t>& input_shape,
		std::vector<size_t>& input_stride,
		size_t offset = 0
	);

	//
	std::tuple<std::vector<size_t>, TensorView, TensorView> do_broadcast(
		const TensorView& other) const;

	TensorView do_mat_alignment() const;

	// getters
	std::vector<size_t> get_shape() const;
	std::vector<size_t> get_stride() const;
	size_t get_offset() const;
	double get_item() const;

	//utils
	friend std::ostream& operator<<(std::ostream& output, const TensorView& view);
	bool is_matrix() const;
	bool is_vector() const;
	bool is_aligned() const;

// protected:
	std::shared_ptr<TensorStorage> m_storage;
	size_t m_offset;
	std::vector<size_t> m_shape;
	std::vector<size_t> m_stride;

};


TensorView generate_tensor(std::vector<size_t> shape, double inital_value);


} //



namespace sic
{
template<typename Fn>
TensorView TensorView::unary_op(
	const Fn& inp_unitary_op) const
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
		double result = inp_unitary_op(get_val(cur_this));
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
	return result_tensor;
}

template<typename Fn>
TensorView TensorView::fold_op(
	const Fn& binary_op,
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

template<typename Fn>
TensorView TensorView::binary_element_wise_op(
	const TensorView& other,
	const Fn& binary_op)
	const
{

	int c_idx = 0;
	int counter = 0;
	size_t cumulative_prod = 1;

	std::vector<double> res_data;
	auto [res_shape, brd_this, brd_other] = do_broadcast(other);
	// std::vector<size_t> res_shape = do_broadcast(other);

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
		double op_result = binary_op(
			get_val(cur_this), other.get_val(cur_other));

		res_tensor.set_val(cur_res, op_result);

		cur_res[lst_sig]++;
		counter++;
		cur_other[lst_sig] = std::min(other_shape[lst_sig] - 1, cur_res[lst_sig]);
		cur_this[lst_sig] = std::min(this_shape[lst_sig] - 1, cur_res[lst_sig]);
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

	return res_tensor;
}

} //
