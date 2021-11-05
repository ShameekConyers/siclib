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
