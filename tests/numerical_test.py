
from pysiclib import *


def test_numerical():
	def example_func(x):
		return x ** 2 + 2 * x - 1

	def no_solution_example(x):
		return x ** 2 + 1

	assert(numerical.equation_solution(example_func, 14) == 3.0)
	assert(numerical.equation_solution(no_solution_example, 0) == None)


	data = linalg.Tensor([x ** 2 + x for x in range(11)])
	assert(numerical.derivative_at_index(data, 5) == 11)

	unit_steps = 100
	data = linalg.Tensor([(x / unit_steps) ** 2 + (x / unit_steps) for x in range(11 * unit_steps)])
	assert(
		abs((numerical.integral_index_interval(data, 0, 5 * unit_steps)/ unit_steps)
		- 50) < 10)



	def system_of_eqs(t, var_arr):
		var_arr = var_arr.get_buffer()[:] #remove this
		dvar_arr = [0, 0]
		dvar_arr[0] = -4 * var_arr[0] + 3 * var_arr[1] + 6
		dvar_arr[1] = 0.6 * dvar_arr[0] - 0.2 * var_arr[1]
		return linalg.Tensor(dvar_arr)

	initial_conditions = linalg.Tensor([0.0, 0.0])

	val = numerical.initial_value_problem(
			system_of_eqs, initial_conditions, 0.5, 0.0)
	val_buffer = val.get_buffer()
	assert(
		abs(val_buffer[0] - 1.79353) < 1e-4 and
		abs(val_buffer[1] - 1.01442) < 1e-4)
