#pragma once

#include "../linalg.hpp"
#include <functional>

namespace sic
{

class ProtoNet {
public:
	ProtoNet(
		size_t num_input_nodes,
		size_t num_hidden_nodes,
		size_t num_hidden_layers,
		size_t num_output_nodes,
		double learning_rate);

	void run_epoch(TensorView input, TensorView target_values);

	TensorView query_net(TensorView input);

// private:

	// shape = (Next Layer size x Previous size)
	std::vector<TensorView> m_weights;
	std::vector<TensorView> m_bias;

	//
	size_t m_num_input_nodes;
	size_t m_num_output_nodes;
	size_t m_num_hidden_nodes;
	size_t m_num_hidden_layers;
	double m_learning_rate;

	// Moving forward remove this
	const std::function<double(double)> m_transform = [](double x)
	{
		// sigmoid
		return (1 / (1 + exp(-x)));
	};
	const std::function<double(double)> m_transform_deriv = [](double x)
	{
		// sigmoid * (1 - sigmoid)
		double val = 1 / (1 + exp(-x));
		return val * (1 - val);
	};
};

}
