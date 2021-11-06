---
nav_name: Models
name: "models"
title: siclib.models
date_added:
date_edited:
description:
---

## A Module based on fitting Models to data.

---

## Documentation

```python
class KNearestNeighbors:
    def __init__(self) -> None: ...
    def fit_model(self, x_vals: _pysiclib.linalg.Tensor, y_vals: _pysiclib.linalg.Tensor, num_neighbors: int, metric: int = ...) -> None: ...
    def predict(self, arg0: _pysiclib.linalg.Tensor) -> _pysiclib.linalg.Tensor: ...

class LinearModel:
    def __init__(self) -> None: ...
    def fit_model(self, x_vals: _pysiclib.linalg.Tensor, y_vals: _pysiclib.linalg.Tensor, fit_procedure: int = ...) -> None: ...
    def predict(self, arg0: _pysiclib.linalg.Tensor) -> _pysiclib.linalg.Tensor: ...

```
