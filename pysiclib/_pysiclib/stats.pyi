from typing import List

import _pysiclib.linalg

def find_kurtosis(arg0: _pysiclib.linalg.Tensor, arg1: int) -> _pysiclib.linalg.Tensor: ...
def find_mean(arg0: _pysiclib.linalg.Tensor, arg1: int) -> _pysiclib.linalg.Tensor: ...
def find_moment(arg0: _pysiclib.linalg.Tensor, arg1: int, arg2: int, arg3: bool, arg4: bool) -> _pysiclib.linalg.Tensor: ...
def find_skew(arg0: _pysiclib.linalg.Tensor, arg1: int) -> _pysiclib.linalg.Tensor: ...
def find_stddev(arg0: _pysiclib.linalg.Tensor, arg1: int) -> _pysiclib.linalg.Tensor: ...
def find_variance(arg0: _pysiclib.linalg.Tensor, arg1: int) -> _pysiclib.linalg.Tensor: ...
def rand_normal_tensor(arg0: float, arg1: float, arg2: List[int]) -> _pysiclib.linalg.Tensor: ...
def rand_uniform_tensor(arg0: float, arg1: float, arg2: List[int]) -> _pysiclib.linalg.Tensor: ...
