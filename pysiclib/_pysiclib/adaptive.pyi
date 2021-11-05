from typing import List

import _pysiclib.linalg

class ProtoNet:
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int, arg4: float) -> None: ...
    def query_net(self, arg0: _pysiclib.linalg.Tensor) -> _pysiclib.linalg.Tensor: ...
    def run_epoch(self, arg0: _pysiclib.linalg.Tensor, arg1: _pysiclib.linalg.Tensor) -> None: ...
    @property
    def m_bias(self) -> List[_pysiclib.linalg.Tensor]: ...
    @property
    def m_weights(self) -> List[_pysiclib.linalg.Tensor]: ...
