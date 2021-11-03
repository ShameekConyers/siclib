from .._pysiclib.adaptive import *
from typing import Any

class ProtoNet_Numpy:
    input_size: Any
    hidden_size: Any
    hidden_layers: Any
    output_size: Any
    learning_rate: Any
    weights: Any
    bias: Any
    func: Any
    def __init__(self, input_size, hidden_size, hidden_layers, output_size, learning_rate, other_net: ProtoNet): ...
    def run_epoch(self, inp, targ) -> None: ...
    def query_net(self, inputs_list): ...
