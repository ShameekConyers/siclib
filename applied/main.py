import numpy as np
import sicnumerical as snum


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

integral = snum.find_integral_from_index(arr, 0, len(arr) - 1) / 4
print(integral)
