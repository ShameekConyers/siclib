#!/bin/bash

/opt/homebrew/anaconda3/bin/python setup.py bdist_wheel &&

/opt/homebrew/anaconda3/bin/python -m pip install dist/sicnumerical-0.0.4-cp38-none-any.whl --force-reinstall
