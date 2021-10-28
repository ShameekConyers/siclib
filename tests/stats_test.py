from pysiclib import *
import numpy as np

def test_stats():
	pass
l =  [0, 1, 2, 3, 4, 5, 0, 2, 4, 6, 8, 10]
my_tensor = linalg.Tensor(l, [2, 3, 2])

print(my_tensor)
x: linalg.Tensor = stats.find_mean(my_tensor, 1)
print(x)


l =  [[0, 1, 2],[ 3, 4, 5]],[[6, 7, 8],[ 9, 10, 11]]
my_tensor = linalg.Tensor(l)
print(my_tensor)
k = my_tensor.transpose()
print(k)


k = my_tensor.transpose()
print(k.get_buffer())
print(hex(id(k.get_buffer)))

input_tensor: linalg.Tensor = linalg.Tensor([[0, 1, 2],[ 3, 4, 5]]).transpose()
print(input_tensor)
mean_tensor = stats.find_moment(input_tensor, 0, 1, False, False)
print(mean_tensor)



l =  [0, 1, 2, 3, 4, 5, 0, 2, 4, 6, 8, 10]
my_tensor = linalg.Tensor(l)
print(my_tensor.get_shape())
# print(stats.find_moment(my_tensor, 0, 2, True, False))
k = [1]
other_tensor = linalg.Tensor(k)
def add(x, y):
	return x + y

print(stats.find_stddev(my_tensor, 0))
v = my_tensor.to_numpy()
print(v.std())


v = [[0, 1], [4, 6], [7, 9]]
l = [[0, 1], [2, 5.6]]

nv = np.array(v)
nl = np.array(l)
v = linalg.Tensor(v)
l = linalg.Tensor(l)

print(v.matmul(l))
print(np.matmul(nv, nl))
