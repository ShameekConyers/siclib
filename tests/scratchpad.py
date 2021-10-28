from pysiclib import *
import numpy as np


l = [-0.10721066,
 -0.08933754,
 -0.08118664,
 -0.09803256,
 -0.08399355,
  0.14557554,
 -0.09055871,
 -0.08826091,
 -0.05405201,
 -0.06772478]
r = [0.9169276, 0.84401998, 0.80108166, 0.79073313, 0.90874806]

x = linalg.Tensor(l).transpose()
y = linalg.Tensor(r).unsqueeze(0)
a = np.array(x.to_numpy())
b = np.array(y.to_numpy())

print(x.matmul(y))
print(np.matmul(a, b))
print(y.get_stride())
print(x.get_stride())
