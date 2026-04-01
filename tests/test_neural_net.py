"""Tests for the neural net module: ProtoNet and ProtoNet_Numpy produce identical outputs."""

import numpy as np
from pysiclib import linalg, adaptive


class TestNeuralNet:
    """Tests for ProtoNet and ProtoNet_Numpy agreement."""

    def test_protonet_and_numpy_agree(self) -> None:
        """Train both nets on the same data and verify identical query outputs."""
        input_size = 3
        hidden_size = 4
        hidden_layers = 1
        output_size = 2
        lr = 0.1

        net = adaptive.ProtoNet(input_size, hidden_size, hidden_layers, output_size, lr)
        np_net = adaptive.ProtoNet_Numpy(
            input_size, hidden_size, hidden_layers, output_size, lr, net
        )

        inputs = [
            [0.2, 0.5, 0.3],
            [0.9, 0.8, 0.1],
            [0.3, 0.1, 0.9],
        ]
        target = [0.0, 1.0]

        for _ in range(20):
            for inp in inputs:
                t_input = linalg.Tensor(inp).transpose()
                t_target = linalg.Tensor(target).transpose()
                net.run_epoch(t_input, t_target)
                np_net.run_epoch(inp, target)

        for inp in inputs:
            t_input = linalg.Tensor(inp).transpose()
            siclib_out = net.query_net(t_input).to_numpy().flatten()
            numpy_out = np_net.query_net(inp).flatten()
            np.testing.assert_array_almost_equal(siclib_out, numpy_out, decimal=6)

    def test_protonet_learns(self) -> None:
        """Verify ProtoNet output moves toward the target after training."""
        net = adaptive.ProtoNet(2, 4, 1, 1, 0.5)
        inp = linalg.Tensor([1.0, 0.0]).transpose()
        target = linalg.Tensor([0.9]).transpose()

        initial_output = net.query_net(inp).get_buffer()[0]

        for _ in range(100):
            net.run_epoch(inp, target)

        trained_output = net.query_net(inp).get_buffer()[0]
        assert abs(trained_output - 0.9) < abs(initial_output - 0.9)
