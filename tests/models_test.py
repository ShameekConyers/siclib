from pysiclib import *


model = models.LinearModel()

data = linalg.Tensor([
	[1, 2],
	[2, 4],
	[6, 5]
])


y_vals = linalg.Tensor([
	2.5,
	5.5,
	8.7,
])

model.fit_model(data, y_vals)

test = linalg.Tensor([
	4, 5
])

prediction = model.predict(test).get_item()
print(prediction)


model = models.KNearestNeighbors()

data = linalg.Tensor([
	[0, 0],
	[1, 1],
	[1, 0],
	[5, 2],
	[2, 4],
	[6, 5]
])


y_vals = linalg.Tensor([
	0.0,
	0.0,
	0.0,
	1.0,
	1.0,
	1.0,
])

model.fit_model(data, y_vals, 2)

test = linalg.Tensor([
	4, 5
])

prediction = model.predict(test).squeeze().get_item()
print(prediction)
