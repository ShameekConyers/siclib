
#include "TensorView.hpp"
#include <iostream>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <OpenBlas/cblas.h>
#endif


// namespace std
// {

// template<>
// struct hash<std::vector<double>> {
// 	size_t operator() (const std::vector<double>& input)
// 	{
// 		size_t bound = std::min(input.size(), (size_t)10);
// 		double counter = 0.0;

// 		for (size_t i = 0; i < bound; i += input.size() / bound) {
// 			counter = input[i] * i * 2.0;
// 		}

// 		return ceil(counter);
// 	}
// };

// template<>
// struct hash<sic::TensorView> {
// 	size_t operator() (const sic::TensorView& input)
// 	{
// 		std::vector<double> v = input.view_buffer();
// 		return (size_t)hash<std::vector<double>>{}(v);
// 	}
// };

// } //

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

template <typename T>
std::ostream& operator<<(std::ostream& output, small_vector<T> const& values)
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

	// TODO format to right, keeping in mind decimal

	for (size_t i = 0; i < this_shape.size(); i++) output << "[";
	while (flag) {

		//! Insert print here

		double val = view.get_val(cur_this);
		output << val;

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
			output << " ";
		}
	}
	output << "Tensor Shape: ";
	output << view.m_shape << "\n";
	return output;
}

TensorView::TensorView()
{
	m_storage = std::make_shared<TensorStorage>(TensorStorage{ {} });
	m_shape = { 0 };
	m_stride = { sizeof(double) };
	m_offset = 0;
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
	const std::vector<double>& input_data,
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
	const TensorView& other_view
)
{
	m_shape = other_view.m_shape;
	m_stride = other_view.m_stride;
	m_offset = other_view.m_offset;
	m_storage = other_view.m_storage;
};

TensorView::TensorView(
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

TensorView& TensorView::operator=(const TensorView& other_view)
{
	m_shape = other_view.m_shape;
	m_stride = other_view.m_stride;
	m_offset = other_view.m_offset;
	m_storage = other_view.m_storage;
	return *this;
}


const std::vector<double>& TensorView::view_buffer() const
{
	return *m_storage;
}

std::vector<double>& TensorView::get_buffer()
{
	return *m_storage;
}

TensorView TensorView::deep_copy() const
{
	return TensorView{ *m_storage, m_shape, m_stride };
}

pybind11::array_t<double> TensorView::to_numpy()
{
	auto stride_in_bytes = m_stride;
	for (auto i = 0; i < m_stride.size(); i++) {
		stride_in_bytes[i] = stride_in_bytes[i] * sizeof(double);
	}
	pybind11::buffer_info result_info(
		m_storage->data(),
		sizeof(double),
		pybind11::format_descriptor<double>::format(),
		m_shape.size(),
		(std::vector<size_t>)m_shape,
		stride_in_bytes
	);
	pybind11::array_t<double> result(result_info);
	return result;
}

}
