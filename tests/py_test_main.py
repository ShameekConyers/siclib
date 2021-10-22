import numpy as np
import numpy
from pysiclib import numerical as snum

arr = np.array([])
for k in range(0, 101):
	arr = np.append(arr, [k ** 2 + k])
new_arr = np.array([])
for count in range(0, 101):
	new_arr = np.append(new_arr, [snum.derivative_at_index(arr, count)])
print(new_arr)

arr = np.array([])
for k in range(0, 401):
	n = k / 4
	arr = np.append(arr, [n ** 2 + n])

arr = np.array([x ** 2 + x for x in range(101)])
integral = snum.integral_index_interval(arr, 0, len(arr) - 1)
print(integral)

def example_func(x):
	return x ** 2 + 2 * x - 1

print(snum.equation_solution(example_func, 14))


data = numpy.array([x ** 2 + x for x in range(11)])
print(snum.derivative_at_index(data, 5))

unit_steps = 100
data = numpy.array([(x / unit_steps) ** 2 + (x / unit_steps) for x in range(11 * unit_steps)])
print(snum.integral_index_interval(data, 0, 5 * unit_steps)/ unit_steps)



def system_of_eqs(t, var_arr):
	dvar_arr = np.zeros(2)
	dvar_arr[0] = -4 * var_arr[0] + 3 * var_arr[1] + 6
	dvar_arr[1] = 0.6 * dvar_arr[0] - 0.2 * var_arr[1]
	return dvar_arr

initial_conditions = np.array([0.0, 0.0])

val = snum.initial_value_problem(
		system_of_eqs, initial_conditions, 0.5, 0.0)
print(val)
