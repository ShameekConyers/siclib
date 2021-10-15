from . import _sicnumerical
import numpy as np

def initial_value_problem(
	system_of_eqs,
	inital_conditions,
	target_value,
	initial_value = 0.0):

	return _sicnumerical.initial_value_problem(
		system_of_eqs,
		inital_conditions,
		target_value,
		initial_value
	)
