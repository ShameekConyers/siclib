from typing import Callable, List

from typing import overload
import numpy

class Tensor:
    @overload
    def __init__(self, numpy_array: numpy.ndarray[numpy.float64]) -> None: ...
    @overload
    def __init__(self, input_data: List[float], input_shape: List[int] = ..., input_stride: List[int] = ..., offset: int = ...) -> None: ...
    @overload
    def __init__(self, other_view: Tensor) -> None: ...
    def binary_element_wise_op(self, arg0: Tensor, arg1: Callable[[float,float],float]) -> Tensor: ...
    def deep_copy(self) -> Tensor: ...
    def dotprod(self, arg0: Tensor) -> Tensor: ...
    def fold_op(self, arg0: Callable[[float,float],float], arg1: float, arg2: int, arg3: bool) -> Tensor: ...
    def get_buffer(self) -> List[float]: ...
    def get_item(self) -> float: ...
    def get_offset(self) -> int: ...
    def get_shape(self) -> List[int]: ...
    def get_stride(self) -> List[int]: ...
    def matinv(self) -> Tensor: ...
    def matmul(self, arg0: Tensor) -> Tensor: ...
    def slice_view(self, arg0: List[int]) -> Tensor: ...
    def squeeze(self, target_dim: int = ...) -> Tensor: ...
    def to_numpy(self) -> numpy.ndarray[numpy.float64]: ...
    def transpose(self, dim_1: int = ..., dim_2: int = ...) -> Tensor: ...
    def unary_op(self, arg0: Callable[[float],float]) -> Tensor: ...
    def unsqueeze(self, arg0: int) -> Tensor: ...
    def view_buffer(self) -> List[float]: ...
