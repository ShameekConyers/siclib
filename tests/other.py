from pysiclib import *
l =  [0, 1, 2, 3, 4, 5, 0, 2, 4, 6, 8, 10]
my_tensor = linalg.Tensor(l, [2, 3, 2])

print(my_tensor)
x: linalg.Tensor = stats.find_mean(my_tensor, 1)
print(x)
