#pragma once

// not used
#include <complex>
#include <vector>
#include <memory>
#include <map>
#include <span>
#include <utility>
#include <functional>



namespace sic
{
struct TensorStorage {
	std::vector<double> m_data;
};

struct TensorView {

	TensorView(
		std::vector<double> input_data,
		std::vector<size_t> input_shape = {},
		std::vector<size_t> input_stride = { 1 },
		size_t offset = 0
	);
	TensorView(
		TensorView& other_view,
		std::vector<size_t> input_shape = {},
		std::vector<size_t> input_stride = { 1 },
		size_t offset = 0
	);
	TensorView deep_copy();
	double get_val(const std::vector<size_t>& selection);
	void set_val(const std::vector<size_t>& selection, double val);

	double& operator[] (size_t element);
	TensorView operator+ (TensorView& other);

	TensorView binary_element_wise_op(TensorView& other,
		std::function<double(double, double)> func
	);

	void update_traverse();
	void set_shape_stride(
		const std::vector<double>& reference_data,
		std::vector<size_t>& input_shape,
		std::vector<size_t>& input_stride,
		size_t offset = 0
	);

	std::shared_ptr<TensorStorage> m_storage;
	size_t m_offset;
	std::vector<size_t> m_shape;
	std::vector<size_t> m_stride;
	std::vector<size_t> m_traverse_offset;
};
}


// namespace end
// {
// 	//using storage = std::span<DataArray>

// 	template<typename T>
// 	struct DataArray {
// 		T* array;
// 		size_t capacity;
// 		size_t current;
// 	};

// 	template<typename T, size_t K = 0>
// 	struct Tensor {

// 		DataArray& data();
// 		Tensor<T, K> deep_copy();
// 		void align();
// 		bool is_aligned();
// 		Tensor<T, K> to_numpy();
// 		Tensor<T, K> to_numpy_transpose();
// 		void to_pytorch();
// 		void to_tensorflow();
// 		void from_pytorch() :
// 			void from_tensorflow();
// 		int save();
// 		void broadcasting();
// 		void permute_shape();
// 		void align_named_shape();
// 		bool is_aligned_name_shape();

// 		std::span<DataArray> data;
// 		size_t offset;
// 		std::vector<size_t> shape;
// 		std::map<std::string, size_t*> named_shape;
// 		std::vector<size_t> stride;
// 		std::vector<size_t> history;
// 		std::vector<std::function<T()>> grad;
// 	};

// 	struct TensorStorage {
// 		std::vector<float> storage;
// 	};

// 	class TensorView {

// 		TensorView getCopy();
// 		TensorView getDeepCopy();
// 		void doAlign();
// 		bool isAligned();
// 		void saveData();
// 		void loadData();

// 		std::shared_ptr<TensorStorage> m_data;
// 		int offset;
// 		std::vector<int> shape;
// 		std::vector<int> stride;
// 	};


// }
