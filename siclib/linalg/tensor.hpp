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

struct TensorStorage {
	std::vector<double> m_data;
};

class TensorObj {

};

class TensorView : public TensorObj {
public:
	TensorView(
		pybind11::array_t<double> numpy_array
	);

	TensorView(
		std::vector<double> input_data,
		std::vector<size_t> input_shape = {},
		std::vector<size_t> input_stride = {},
		size_t offset = 0
	);

	TensorView(
		const TensorView& other_view
	)
	{
		m_shape = other_view.m_shape;
		m_stride = other_view.m_stride;
		m_offset = other_view.m_offset;
		m_storage = other_view.m_storage;
	};

	TensorView(
		const TensorView& other_view,
		std::vector<size_t> input_shape,
		std::vector<size_t> input_stride,
		size_t offset
	)
	{
		m_shape = input_shape;
		m_stride = input_stride;
		m_offset = offset;
		m_storage = other_view.m_storage;
	};

	TensorView& operator=(const TensorView& other_view)
	{
		m_shape = other_view.m_shape;
		m_stride = other_view.m_stride;
		m_offset = other_view.m_offset;
		m_storage = other_view.m_storage;
		return *this;
	}

	// TensorView(TensorView&& other) = default;
	// TensorView& operator=(TensorView&& other) = default;
	const	std::vector<double>& view_buffer() const;
	std::vector<double>& get_buffer();

	TensorView deep_copy() const;
	pybind11::array_t<double> to_numpy();

	TensorView slice_view(const std::vector<ssize_t>& selection) const;
	double get_val(const std::vector<size_t>& selection) const;
	void set_val(const std::vector<size_t>& selection, double val);


	double& operator[] (size_t element);
	TensorView operator+ (TensorView& other);
	TensorView operator- (TensorView& other);

	TensorView unitary_op(
		const std::function<double(double)>& func
	);
	TensorView binary_element_wise_op(const TensorView& other,
		const std::function<double(double, double)>& func
	) const;

	TensorView fold_op(
		const std::function<double(double, double)>& binary_op,
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

	TensorView do_alignment() const; // TODO

	// getters
	std::vector<size_t> get_shape() const;
	std::vector<size_t> get_stride() const;
	size_t get_offset() const;
	double get_item() const;

	//utils
	friend std::ostream& operator<<(std::ostream& output, const TensorView& view);
	bool is_matrix() const;
	bool is_vector() const;
	bool is_aligned() const; // TODO

// protected:
	std::shared_ptr<TensorStorage> m_storage;
	size_t m_offset;
	mutable std::vector<size_t> m_shape;
	mutable std::vector<size_t> m_stride;

};


TensorView generate_tensor(std::vector<size_t> shape, double inital_value);

}
