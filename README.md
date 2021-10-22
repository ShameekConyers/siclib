# siclib

A library of various things - with a focus on math.

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
A Collection of various Numerical approximation methods, uses numpy currently

### Linalg
A collection of various Linear Algebra operations and structures
