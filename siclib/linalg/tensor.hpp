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
		const TensorView& other_view,
		std::vector<size_t> input_shape = {},
		std::vector<size_t> input_stride = {},
		size_t offset = 0
	);



	std::vector<double>& get_buffer();

	TensorView deep_copy();
	pybind11::array_t<double> to_numpy();

	double get_val(const std::vector<size_t>& selection) const;
	void set_val(const std::vector<size_t>& selection, double val);


	double& operator[] (size_t element);
	TensorView operator+ (TensorView& other);
	TensorView operator- (TensorView& other);


	TensorView binary_element_wise_op(TensorView& other,
		std::function<double(double, double)> func
	);

	TensorView apply_op(
		std::vector<std::function<double(double, double)>> function_vector,
		size_t target_dim,
		double other
	);

	TensorView fold_op(
		std::function<double(double, double)> function_vector,
		double inital_value,
		size_t target_dim,
		bool left_op = true
	);

	void set_shape_stride(
		const std::vector<double>& reference_data,
		std::vector<size_t>& input_shape,
		std::vector<size_t>& input_stride,
		size_t offset = 0
	);

	void squeeze();
	void unsqueeze();
	void transpose();

	std::vector<size_t> do_broadcast(TensorView& other);
	void undo_broadcast();


	// getters
	std::vector<size_t> get_shape();
	std::vector<size_t> get_stride();
	size_t get_offset();

	//utils
	friend std::ostream& operator<<(std::ostream& output, const TensorView& view);

private:
	std::shared_ptr<TensorStorage> m_storage;
	size_t m_offset;
	mutable std::vector<size_t> m_shape;
	mutable std::vector<size_t> m_stride;

	mutable std::vector<size_t> m_saved_shape;
	mutable std::vector<size_t> m_saved_stride;
};


TensorView generate_tensor(std::vector<size_t> shape, double inital_value);

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
