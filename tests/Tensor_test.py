import pysiclib
from pysiclib import numerical
from pysiclib.api.numerical import derivative_at_index
import numpy as np
l = np.array([[[1, 2], [4, 5]], [[1, 2], [4, 5]]])
# l = np.array([1])
print(l.shape)
l = l.reshape([1,1, 2, 2, 1, 2, 1])
print(l.shape)
print(l.strides)
print()
l =l.transpose()
l = l.reshape([1, 1, 1, 2, 1, 2, 2, 1, 1])
# l = l.reshape([1,1, 1, 1, 1, 1])
print(l.shape)
print(l.strides)
print()
l = np.array([])
print(l.shape)
print(l.strides)
l = np.array([[[1, 2], [4, 5]], [[10, 2], [4, 5]]])
r = pysiclib.linalg.Tensor(l)
print(l.strides)
print(r.get_stride())
print(r.get_buffer()[0])
print(r.get_buffer()[4])

l= np.ones([2, 2])
k = np.ones([3, 2, 2])
print(l.strides)
print(l.shape)
print()


print(l)
v1 = pysiclib.linalg.Tensor(l)
v2 = pysiclib.linalg.Tensor(l)
def add(r, l):
	return r + l
v3 = pysiclib.linalg.Tensor(v2)
v4 = v2.binary_element_wise_op(v1, add)
print(v4.to_numpy())
# res = v4.get_buffer()
# print(res)
# v2 = v1.binary_element_wise_op(v1, add)
# print(id(v3.get_buffer()))
# print(id(v2.get_buffer()))
# print(v1.get_buffer())

# def _add(k, i):
# 	return add(i + 100, 3)

# def derivative_at_index(*args):
# 	return numerical.derivative_at_index(*args)


# arr = np.array([])
# for k in range(0, 101):
# 	arr = np.append(arr, [k ** 2 + k])
# new_arr = np.array([])
# for count in range(0, 101):
# 	new_arr = np.append(new_arr, [derivative_at_index(arr, count)])
# print(new_arr)


l = np.array([[1, 2], [3, 4]])
r = pysiclib.linalg.Tensor(l)
rinv = r.matinv()
print(rinv)
