---
nav_name: Adaptive
name: "adaptive"
title:  siclib.Adaptive
date_added:
date_edited:
description:
---

## A Module based on Adaptive Models.

## Implementation Notes

Tensor operations are unoptimized. I recommend PyTorch or TorchLib instead for a
general Tensor/NN Library.


---

## Documentation

```python
class ProtoNet:
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int, arg4: float) -> None: ...
    def query_net(self, arg0: _pysiclib.linalg.Tensor) -> _pysiclib.linalg.Tensor: ...
    def run_epoch(self, arg0: _pysiclib.linalg.Tensor, arg1: _pysiclib.linalg.Tensor) -> None: ...
```
