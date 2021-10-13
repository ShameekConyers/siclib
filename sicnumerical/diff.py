from . import _sicnumerical

def find_derivative_from_index(array, index):
	return _sicnumerical.diff_from_index(array, index)

def find_derivative_from_value(array, index):
	return _sicnumerical.diff_from_value(array, index)
