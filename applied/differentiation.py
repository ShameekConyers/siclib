import numpy as np
import sicnumerical as snum


arr = np.array([])
for k in range(0, 101):
	arr = np.append(arr, [k ** 2 + k])
new_arr = np.array([])
for count in range(0, 101):
	new_arr = np.append(new_arr, [snum.diff(arr, count)])
print(new_arr)
