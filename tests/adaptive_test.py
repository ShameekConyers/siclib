from pysiclib import *
import numpy as np

r = adaptive.ProtoNet(3, 2, 1, 2, 1)

i = linalg.Tensor([0.2, 0.5, 0.3]).transpose()
f = linalg.Tensor([0.9, 0.8, 0.1]).transpose()
k = linalg.Tensor([0.3, 0.1, 0.9]).transpose()
o = linalg.Tensor([0, 1]).transpose()
print(i)
print(o)
for j in range(10):
	r.run_epoch(i, o)
	r.run_epoch(f, o)
	r.run_epoch(k, o)

print(r.query_net(i))


test_data = None
with open("data/large/mnist_test.csv", "r") as f:
	test_data = f.readlines()

train_data = None
with open("data/large/mnist_train.csv", "r") as f:
	train_data = f.readlines()

input_nodes = 784
hidden_nodes = 50
output_nodes = 10
learning_rate = 0.1

my_net = adaptive.ProtoNet(
	input_nodes, hidden_nodes, 1, output_nodes, learning_rate)
np_net = adaptive.ProtoNet_Numpy(
	input_nodes, hidden_nodes, 1, output_nodes, learning_rate, my_net)

counter = 0
score = []
nscore = []

for record in train_data[:200]:
	if(counter % 100 == 0):
		print("{}...".format(counter))
	counter += 1

	all_values = record.split(',')

	scaled_input_raw = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
	scaled_input = linalg.Tensor(scaled_input_raw).transpose()

	scaled_target_raw = np.zeros(output_nodes) + 0.01
	scaled_target_raw[int(all_values[0])] = 0.99

	scaled_target = linalg.Tensor(scaled_target_raw).transpose()


	my_net.run_epoch(scaled_input, scaled_target)
	np_net.run_epoch(scaled_input_raw, scaled_target_raw)

	correct_label = int(all_values[0])

	query_res = my_net.query_net(scaled_input)
	numpy_query = np_net.query_net(scaled_input_raw)

	label = np.argmax(query_res.to_numpy())
	numpy_label = np.argmax(numpy_query)

	if label == correct_label:
		score.append(1)
	else:
		score.append(0)

	if numpy_label == correct_label:
		nscore.append(1)
	else:
		nscore.append(0)


print("------")
print(sum(score) / len(score))
print("------")
print(sum(nscore) / len(nscore))
print("\n\n\n\n\n")


score = []
test_counter = 0
for record in test_data[:100]:
	test_counter += 1

	all_values = record.split(',')

	scaled_input_raw = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
	scaled_input = linalg.Tensor(scaled_input_raw).transpose()

	correct_label = int(all_values[0])

	query_res = my_net.query_net(scaled_input)
	numpy_query = np_net.query_net(scaled_input_raw)
	# print(query_res.squeeze())
	label = np.argmax(query_res.to_numpy())
	numpy_label = np.argmax(numpy_query)
	###	TEST CASE BELOW
	assert(label == numpy_label)

	if label == correct_label:
		score.append(1)
	else:
		score.append(0)
	if test_counter == 30:
		break

print("------")
print(sum(score) / len(score))
print("\n\n\n\n\n")


# for i in range(100):
# 	n = i / 100
# 	print(expit(n))
# 	print(my_net.m_transform(n))
# 	print("_____")
