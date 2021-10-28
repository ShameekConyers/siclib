# siclib

A C++ library of various things - with a focus on math.

pysiclib is the python interface.

## Corresponding Project and Documentation
Can be found here <a href ="https://shameekconyers.com/projects/siclib">here</a>

## pysiclib -  Python Installation Instructions
<!-- ```shell
pip install sicnumerical
``` -->
Make sure you have CMake installed with an appropriate compiler, then make sure
you working directory is the same as the root of this project
```shell
$ pip install -r requirements.txt &&
 python setup.py bdist_wheel &&
 pip install dist/*

$ python
>>> import pysiclib
```


In the future I will have the project uploaded to pypi so you can install from
just pip.

## Current Modules

### Numerical
A collection of various Numerical approximation methods

### Linalg
A collection of various Linear Algebra operations and structures

### Stats
A collection of various procedures from statistics

### Adaptive (name pending)

A collection of adaptive learning methods.
