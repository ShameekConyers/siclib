#include "NeuralNet.hpp"
#include "../stats.hpp"
#include <cassert>
#include <exception>
#include <deque>
#include <iostream>

namespace sic
{

ProtoNet::ProtoNet(size_t num_input_nodes,
	size_t num_hidden_nodes,
	size_t num_hidden_layers,
	size_t num_output_nodes,
	double learning_rate)
{
	m_num_input_nodes = num_input_nodes;
	m_num_hidden_nodes = num_hidden_nodes;
	m_num_hidden_layers = num_hidden_layers;
	m_num_output_nodes = num_hidden_nodes;
	m_learning_rate = learning_rate;

	// initialize with random tensor views

	size_t cur_nodes = num_input_nodes;
	for (size_t i = 0; i < num_hidden_layers; i++) {

		m_weights.push_back(
			rand_uniform_tensor(-0.5, 0.5, { num_hidden_nodes, cur_nodes }));
		m_bias.push_back(
			rand_uniform_tensor(-0.5, 0.5, { num_hidden_nodes, 1 }));
		cur_nodes = num_hidden_nodes;
	}
	m_weights.push_back(
		rand_uniform_tensor(-0.5, 0.5, { num_output_nodes, cur_nodes }));
	m_bias.push_back(
		rand_uniform_tensor(-0.5, 0.5, { num_output_nodes, 1 }));
}

void ProtoNet::run_epoch(TensorView input, TensorView target_values)
{
	if (input.get_shape().size() != 2 && input.get_shape()[1] != 1) {
		throw std::runtime_error("input should be of shape Ix1");
	};
	if (target_values.get_shape().size() != 2
		&& target_values.get_shape()[1] != 1) {
		throw std::runtime_error("target should be of shape Tx1");
	};

	std::vector<TensorView> output_vec;
	std::vector<TensorView> raw_output_vec;
	std::deque<TensorView> error_vec;

	// forward
	TensorView layer_input = input;
	TensorView layer_output = input;
	for (size_t i = 0; i < m_weights.size(); i++) {
		layer_output = m_weights[i]
			.matmul(layer_input);
		raw_output_vec.push_back(layer_output); // hi
		layer_input = layer_output.unitary_op(m_transform);
		output_vec.push_back(layer_input); // ho
	}

	std::function <double(double)> func = [](double x)
	{
		return -2 * x;
	};

	// backprop error
	error_vec.push_front(target_values - output_vec.back());
		// .unitary_op(func));
	for (ssize_t i = m_weights.size() - 2; i >= 0; i--) {
		// error
		TensorView layer_error = m_weights[i + 1]
			.transpose()
			.matmul(error_vec.front());
			// .binary_element_wise_op(
			// 	raw_output_vec[i].unitary_op(m_transform_deriv),
			// 	std::multiplies());

		error_vec.push_front(layer_error);
	}

	// update
	double lr = m_learning_rate;
	std::function<double(double)> learn_func = [lr](double x)
	{
		return lr * x;
	};

	for (ssize_t i = m_weights.size() - 1; i >= 0; i--) {

		// dW = (from x to) ~ (j, l)
		TensorView target = input.transpose();
		if (i != 0) {
			target = output_vec[i - 1].transpose();
		}

		TensorView delta_weights = error_vec[i]
			.binary_element_wise_op(
				raw_output_vec[i].unitary_op(m_transform_deriv), std::multiplies())
			.matmul(target)
			.unitary_op(learn_func);
		TensorView delta_bias = error_vec[i].unitary_op(learn_func);

		// if (i != 0) {
		// 	std::cerr << "YOU:\n";
		// 	std::cerr << error_vec[i]
		// 		.binary_element_wise_op(
		// 			raw_output_vec[i].unitary_op(m_transform_deriv), std::multiplies());

		// 	std::cerr << target;

		// 	std::cerr <<
		// 		error_vec[i]
		// 		.binary_element_wise_op(
		// 			raw_output_vec[i].unitary_op(m_transform_deriv), std::multiplies())
		// 		.matmul(target);

		// 	std::cerr << "\n";
		// }

		m_weights[i] = m_weights[i] + delta_weights;
		m_bias[i] = m_bias[i] + delta_bias;
	}


	// std::cerr << "-------\n";
	// std::cerr << target_values - output_vec.back();
	// std::cerr << target_values;
	// std::cerr << output_vec.back();
	// std::cerr << "-------\n";
}

TensorView ProtoNet::query_net(TensorView input)
{
	if (input.get_shape().size() != 2 && input.get_shape()[1] != 1) {
		throw std::runtime_error("input should be of shape Ix1");
	};
	TensorView layer_input = input;
	TensorView layer_output = input; // Temp
	for (size_t i = 0; i < m_weights.size(); i++) {
		layer_output = m_weights[i]
			.matmul(layer_input)
			// .binary_element_wise_op(m_bias[i], std::plus())
			.unitary_op(m_transform);
		layer_input = layer_output;
	}

	return layer_output;
}

}
