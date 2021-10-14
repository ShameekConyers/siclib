import numpy as np
import sicnumerical as snum
import numpy
import sicnumerical

arr = np.array([])
for k in range(0, 101):
	arr = np.append(arr, [k ** 2 + k])
new_arr = np.array([])
for count in range(0, 101):
	new_arr = np.append(new_arr, [snum.find_derivative_from_index(arr, count)])
print(new_arr)

arr = np.array([])
for k in range(0, 401):
	n = k / 4
	arr = np.append(arr, [n ** 2 + n])

arr = np.array([x ** 2 + x for x in range(101)])
integral = snum.find_integral_from_index(arr, 0, len(arr) - 1)
print(integral)

def example_func(x):
	return x ** 2 + 2 * x - 1

print(snum.find_equation_solution(example_func, 14))


data = numpy.array([x ** 2 + x for x in range(11)])
print(sicnumerical.find_derivative_from_index(data, 5))

data = numpy.array([x ** 2 + x for x in range(11)])
print(sicnumerical.find_integral_from_index(data, 6, 11))

x, y = 4, 6
print(6)