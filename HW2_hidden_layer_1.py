import numpy as np
import os.path
import math
import tensorflow as tf

N = 4000
INPUT_SIZE = 2
LABELS = 2
SPLIT = [0.7, 0.2, 0.1]

batch_size = 10
hidden_cells_1 = 10
hidden_cells_2 = 10
learning_rate = 0.5
training_epochs = 100
display_step = 10

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
	return (100.0 * np.sum(np.argmax(pred, 1) == np.argmax(label, 1)) / pred.shape[0])
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
def init_variables_1_hidden_layer(input_size, num_labels, num_hidden1):

	# hidden layer 1
	w1 = tf.Variable(tf.truncated_normal([input_size, num_hidden1], stddev=0.1))
	b1 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))
	
	# output layer
	w2 = tf.Variable(tf.truncated_normal([num_hidden1, num_labels], stddev=0.1))
	b2 = tf.Variable(tf.constant(1.0, shape = [num_labels]))

	variables = {
		'w1': w1, 'w2': w2,
		'b1': b1, 'b2': b2
	}
	return variables

def model_1_hidden_layer(data, variables):

	layer1_fccd = tf.matmul(data, variables['w1']) + variables['b1']
	layer1_actv = tf.nn.relu(layer1_fccd)

	layer2_fccd = tf.matmul(layer1_actv, variables['w2']) + variables['b2']
	logits = tf.nn.sigmoid(layer2_fccd)

	return logits

# save_path = '/Users/yankong/Documents/Tensorflow/ECE283_HW2/'
# name = os.path.join(save_path, 'result.txt')
# file = open(name, 'w')

graph = tf.Graph()

# for learning_rate in LR:
# 	for batch_size in BATCH_SIZE:
# 		for hidden_cells_1 in HIDDEN_CELLS_1:
# 			for training_epochs in TRAINING_EPOCHS:
print('Learning Rate = {:03f}, Batch Size = {:d}, Cells for hidden layer1: {:d}, Training epochs:{:d} '.format(learning_rate, batch_size, hidden_cells_1, training_epochs))
# file.write('Learning Rate = {:03f}, Batch Size = {:d}, Cells for hidden layer1: {:d}, Training epochs:{:d} '.format(learning_rate, batch_size, hidden_cells_1, training_epochs))
with graph.as_default():
	tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, INPUT_SIZE))
	tf_train_label = tf.placeholder(tf.float32, shape = (batch_size, LABELS))
	tf_validation_dataset = tf.constant(validation_set, tf.float32)
	tf_test_dataset = tf.constant(test_set, tf.float32)
	# initilization of weight and bias
	variables = init_variables_1_hidden_layer(INPUT_SIZE, LABELS, hidden_cells_1)
	# initialize model
	logits = model_1_hidden_layer(tf_train_dataset, variables)
	# loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = tf_train_label))
	# optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
	# prediction for training and test data
	train_pred = logits

	validation_pred = model_1_hidden_layer(tf_validation_dataset, variables)
	test_pred = model_1_hidden_layer(tf_test_dataset, variables)

with tf.Session(graph = graph) as session:
	tf.global_variables_initializer().run()
	for epoch in range(training_epochs):
		avg_cost = 0.0
		total_batch = int(len(training_set) / batch_size)
		X_batches = np.array_split(training_set, total_batch)
		Y_batches = np.array_split(training_label, total_batch)
		print('-------------------------- epoch {:04d} --------------------------'.format(epoch))
		for i in range(total_batch):
			data, label = X_batches[i], Y_batches[i]
			_, cost, pred = session.run([optimizer, loss, train_pred], feed_dict =  {tf_train_dataset: data, tf_train_label:label})
			train_accuracy = accuracy(pred, label)
			if i % display_step == 0:
				print('step {:04d} : cost is {:.6f}, accuracy on training set {:02.1f} %'.format(i, cost, train_accuracy))
	print('Training Finished!')

	validation_accuracy = accuracy(validation_pred.eval(), validation_label)

	test_accuracy = accuracy(test_pred.eval(), test_label)
	print('Validation set Accuracy:', validation_accuracy)
	print('Test set Accuracy:', test_accuracy)


	# file.write('Accuracy of Validation set: ' + str(validation_accuracy) + '\n')

#file.close()

