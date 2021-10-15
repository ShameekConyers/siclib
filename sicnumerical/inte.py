from . import _sicnumerical

def integral_index_interval(
	input_array, start_index = 0, end_index = None):
	if end_index == None:
		end_index = len(input_array)
	return _sicnumerical.integral_index_interval(
		input_array, start_index, end_index)
