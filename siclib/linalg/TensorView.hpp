#pragma once

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
	); // TODO

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

	TensorView transpose(ssize_t dim_1 = -1, ssize_t dim_2 = -1) const;

	// matrix methods
	TensorView matmul(const TensorView& other) const;
	TensorView matinv() const;

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

	// transforms
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

	TensorView switch_mat_major_order() const;

// protected:
	std::shared_ptr<TensorStorage> m_storage;
	size_t m_offset;
	std::vector<size_t> m_shape;
	std::vector<size_t> m_stride;

};


TensorView generate_tensor(std::vector<size_t> shape, double inital_value);


} //


#include "TensorView.tpp"
