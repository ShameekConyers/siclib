from . import _sicnumerical

def equation_solution(equation_as_function, target_val, precision = 6):
	output_val =\
		_sicnumerical.equation_solution(equation_as_function, target_val)
	if (output_val == None):
		 return None
	else:
		return round(output_val, precision)
