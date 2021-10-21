from . import _pysiclib

#differentiation

def derivative_at_index(input_array, index):
	return _pysiclib.derivative_at_index(input_array, index)

def derivative_at_value(input_array, value):
	return _pysiclib.derivative_at_value(input_array, value)

#intial value
def initial_value_problem(
	system_of_eqs,
	inital_conditions,
	target_value,
	initial_value = 0.0):

	return _pysiclib.initial_value_problem(
		system_of_eqs,
		inital_conditions,
		target_value,
		initial_value
	)

#integration
def integral_index_interval(
	input_array, start_index = 0, end_index = None):
	if end_index == None:
		end_index = len(input_array)
	return _pysiclib.integral_index_interval(
		input_array, start_index, end_index)

#sol equations
def equation_solution(equation_as_function, target_val, precision = 6):
	output_val =\
		_pysiclib.equation_solution(equation_as_function, target_val)
	if (output_val == None):
		 return None
	else:
		return round(output_val, precision)
