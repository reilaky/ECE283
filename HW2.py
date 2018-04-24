import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf

N = 4000
INPUT_SIZE = 2
LABELS = 2
BATCH_SIZE = 10
HIDDEN_CELLS_1 = 10
HIDDEN_CELLS_2 = 10
SPLIT = [0.7, 0.2, 0.1]
LR = 0.1

training_epochs = 1000
display_step = 1

def generateGuassian(mean, theta, lambd, num):
	u1 = np.array([[np.cos(theta)], [np.sin(theta)]])
	u2 = np.array([[-np.sin(theta)], [np.cos(theta)]])
	cov = lambd[0] * np.dot(u1, np.transpose(u1)) + lambd[1] * np.dot(u2, np.transpose(u2))
	sample = np.random.multivariate_normal(mean, cov, num)
	return sample, cov

def reformat_data(dataset, label):
	num_label = len(np.unique(label))
	np_label_ = one_hot_encode(np.array(label, dtype=np.float32), num_label)
	np_dataset, np_label = randomize(dataset, np_label_)
	return np_dataset, np_label

def randomize(dataset, label):
	permutation = np.random.permutation(label.shape[0])
	shuffled_dataset = dataset[permutation, :]
	shuffled_label = label[permutation]
	return shuffled_dataset, shuffled_label

def one_hot_encode(np_array, num_label):
	temp = (np.arange(num_label) == np_array[:,None]).astype(np.float32)
	return temp

def accuracy(pred, label):
	return (100.0 * np.sum(np.argmax(pred, 1) == np.argmax(label, 1)) / pred.shape[0])

def split_dataset(dataset, label, split):
	training_set = np.array(dataset[:int(N * split[0]), :])
	training_label = label[:int(N * split[0]), :]
	validation_set = np.array(dataset[int(N * split[0]): round(N * (split[0] + split[1])), :])
	validation_label = label[int(N * split[0]): round(N * (split[0] + split[1])),:]
	test_set = np.array(dataset[round(N * (split[0] + split[1])): round(N * (split[0] + split[1] + split[2])), :])
	test_label = label[round(N * (split[0] + split[1])): round(N * (split[0] + split[1] + split[2])), :]
	print('The size of training set is',np.shape(training_set))
	print('The size of validation set is',np.shape(validation_set))
	print('The size of test set is', np.shape(test_set))
	return training_set, training_label, validation_set, validation_label, test_set, test_label

def get_dataset(num):
	# for Class0
	mean = [0, 0]
	theta = 0
	lambd = [2, 1]
	sample0, cov = generateGuassian(mean, theta, lambd, num)
	print('Class0 cov = \n', cov)
	Y0 = np.empty(num)
	Y0.fill(0)

	# for Class1
	meanA = [-2, 1]
	piA = 1 / 3
	thetaA = - np.pi * 3 / 4
	lambdaA = [2, 1/4]
	sample1A, covA = generateGuassian(meanA, thetaA, lambdaA, round(num * piA))
	print('Class1 covA = \n', covA)

	meanB = [3, 2]
	piB = 2 / 3
	thetaB = np.pi / 4
	lambdaB = [3, 1]
	sample1B, covB = generateGuassian(meanB, thetaB, lambdaB, round(num * piB))
	print('Class1 covB = \n', covB)
	sample1 = np.concatenate((sample1A, sample1B), axis = 0)
	Y1 = np.empty(num)
	Y1.fill(1)

	dataset = np.concatenate((sample0, sample1), axis = 0)
	label = np.concatenate((Y0, Y1), axis = 0)
	return dataset, label

def accuracy(pred, label):
	return (np.sum(pred == label))
# ---------------------------------- data processing -------------------------------------
raw_dataset, raw_label = get_dataset(N // 2)
dataset, label = reformat_data(raw_dataset, raw_label)
training_set, training_label, validation_set, validation_label, test_set, test_label = split_dataset(dataset, label, SPLIT)

# ---------------------------------- nerual network --------------------------------------
'''
Acitivation function: ReLU nonlinearities in the layers and a sigmoid at the output, 
Loss function: binary cross entropy as the loss function. 
Play with the number of neurons in each layer, and use L2 weight regularization, 
tuning these parameters to get “adequate” performance on the validation set.

'''
# 1 input layer, 1/2 hidden layer, 1 output layer
X = tf.placeholder(dtype = tf.float32, shape = [None, INPUT_SIZE])
Y = tf.placeholder(dtype = tf.float32, shape = [None, LABELS])


# sess.run(***, feed_dict={input: **}).
def add_layer(inputs, input_size, output_size, activation_function = None):
	weights = tf.Variable(tf.random_normal([input_size, output_size]))
	biases = tf.Variable(tf.random_normal([output_size]))
	fccd = tf.matmul(inputs, weights) + biases
	if activation_function is None:
		outputs = fccd
	else:
		outputs = activation_function(fccd)
	return outputs

# add hidden layer1
hidden_l = add_layer(X, INPUT_SIZE, HIDDEN_CELLS_1, activation_function = tf.nn.relu)
# add hidden layer2
# l2 = add_layer(l1, HIDDEN_CELLS_1, HIDDEN_CELLS_2)
# add output layer
output_l = add_layer(hidden_l, HIDDEN_CELLS_1, LABELS, activation_function = tf.nn.sigmoid)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = output_l, labels = Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= LR).minimize(loss)
 
sess = tf.Session()
init = tf.global_variables_initializer()


with tf.Session() as sess:
	sess.run(init)
	for epoch in range(training_epochs):
		avg_cost = 0.0
		total_batch = len(training_set) // BATCH_SIZE
		X_batches = np.array_split(training_set, total_batch)
		Y_batches = np.array_split(training_label, total_batch)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = X_batches[i], Y_batches[i]
			# Run optimization op (backprop) and cost op (to get loss value)
			_, pred,cost = sess.run([optimizer, output_l, loss], feed_dict={X: batch_x, Y: batch_y})
			# Compute average loss
			avg_cost += cost / total_batch
			# Display logs per epoch step
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
			# Calculate accuracy
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.5f}".format(avg_cost))
	print("Optimization Finished!")

# for step in range(steps):
# 	offset = (step * BATCH_SIZE) % (training_label.shape[0] - BATCH_SIZE)
# 	data = training_set[offset:(offset + BATCH_SIZE), :]
# 	label = training_label[offset:(offset + BATCH_SIZE)]
# 	# print('label',label)
# 	_, cost = sess.run([optimizer, loss], feed_dict = {X: data, Y: label})
	
# 	# train_accuracy = accuracy(pred, label)
# 	if step % 2 == 0:
# 		summary = "step {:04d} : loss is {:f}".format(step, cost)
# 		print(summary)









