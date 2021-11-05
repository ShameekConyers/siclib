# siclib

A C++ library of various things - with a focus on math.

pysiclib is the python interface.

## Corresponding Project and Documentation
Can be found here <a href ="https://shameekconyers.com/projects/siclib">here</a>

## pysiclib -  Python Installation Instructions

Make sure you have Andaconda and a C++ compiler installed on your system.
If you don't have an Apple computer make sure you have installed OpenBlas.

```shell
$ conda install pybind11 &&
  git clone https://github.com/ShameekConyers/siclib &&
  cd siclib &&
  pip install -r requirements.txt &&
  python setup.py bdist_wheel &&
  pip install dist/pysiclib-0.0.6-cp38-none-any.whl

$ python
>>> import pysiclib
```

Even though I have this uploaded to PyPi so you can install from
just pip, i've been having issues with cross-platform getting builds
working proprly - until then above is the most straightforward way

## Current Modules

### Numerical
A collection of various Numerical approximation methods

### Linalg
A collection of various Linear Algebra operations and structures

### Stats
A collection of various procedures from statistics

### Adaptive (name pending)

A collection of adaptive learning methods.

### Models

A collection of models for fitting data.
