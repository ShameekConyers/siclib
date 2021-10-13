from . import _sicnumerical

def find_integral_from_index(input_array, start_index = 0, end_index = None):
	if end_index == None:
		end_index = len(input_array)
	return _sicnumerical.find_integral_from_index(input_array, start_index, end_index)
