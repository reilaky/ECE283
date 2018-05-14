import matplotlib.pyplot as plt
from EM import *
from Kmeans import *

N = 200

def generateGuassian(mean, theta, lambd, num):
	u1 = np.array([[np.cos(theta)], [np.sin(theta)]])
	u2 = np.array([[-np.sin(theta)], [np.cos(theta)]])
	cov = lambd[0] * np.dot(u1, np.transpose(u1)) + lambd[1] * np.dot(u2, np.transpose(u2))
	sample = np.random.multivariate_normal(mean, cov, num)
	return sample, cov

def plot_samples(pos, sample):
	plt.subplot(2, 2, pos)
	color = ['C0', 'C1', 'C2', 'C3', 'C4']
	i = 1
	for s in sample:
		plt.scatter(s[:, 0], s[:, 1], s = 2, c = color[i - 1], label = 'Component' + str(i))
		i += 1

def plot_boundary(sample, model, K):
	h = .02
	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = sample[:, 0].min() - 1, sample[:, 0].max() + 1
	y_min, y_max = sample[:, 1].min() - 1, sample[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Obtain labels for each point in mesh. Use last trained model.
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.imshow(Z, interpolation='nearest',
	           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	           cmap=plt.cm.Paired,
	           aspect='auto', origin='lower')

	plt.title('K =' + str(K))
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())

def one_hot_encode(np_array, num_label):
	temp = (np.arange(num_label) == np_array[:,None]).astype(np.float32)
	return temp

def reformat_data(dataset, label):
	num_label = len(np.unique(label))
	np_label = one_hot_encode(np.array(label, dtype=np.float32), num_label)
	return dataset, np_label

def generate_ortho_vec(prob, num, dimen, distr, thres = 1):
	random_vec = np.random.choice(distr, size = [num, dimen], p = prob)
	orthogonal = False
	diag_indices = np.diag_indices(num)
	n = 0
	while (orthogonal == False):
		dot_prod = np.dot(random_vec, random_vec.T)
		dot_prod[diag_indices] = 0
		correlated = np.sum(np.absolute(dot_prod), axis = 1)
		if((correlated == 0).all()) or (np.sum(correlated) < thres):
			orthogonal = True
		else:
			n += 1
			purge_index = np.argmax(correlated)
			random_vec[purge_index,:] = np.random.choice(distr, dimen, p = prob)
	return random_vec

pi = [1/2, 1/6, 1/3]
# component 1:
mean1 = [0, 0]
theta1 = 0
lambd1 = [2, 1]
sample1, cov1 = generateGuassian(mean1, theta1, lambd1, round(N * pi[0]))
label1 = np.empty(sample1.shape[0])
label1.fill(0)
print('Class0 cov = \n', cov1)

# component 2
mean2 = [-2, 1]
theta2 = - np.pi * 3 / 4
lambda2 = [2, 1/4]
sample2, cov2 = generateGuassian(mean2, theta2, lambda2, round(N * pi[1]))
label2 = np.empty(sample2.shape[0])
label2.fill(1)
print('Class1 covA = \n', cov2)

# component 3
mean3 = [3, 2]
theta3 = np.pi / 4
lambda3 = [3, 1]
sample3, cov3 = generateGuassian(mean3, theta3, lambda3, round(N * pi[2]))
label3 = np.empty(sample3.shape[0])
label3.fill(2)
print('Class1 covB = \n', cov3)

sample = np.concatenate((sample1, sample2, sample3), axis = 0)
label = np.concatenate((label1, label2, label3), axis = 0)
prob_z = np.array([len(label1), len(label2), len(label3)]).reshape(3, 1) / N
sample, z = reformat_data(sample, label)
ground_truth = np.argmax(z, axis = 1)
plt.figure(figsize = (20, 10))
plt.title('Data Sample')
plt.scatter(sample1[:, 0], sample1[:, 1], s = 2, c = 'r', label = 'Component 1')
plt.scatter(sample2[:, 0], sample2[:, 1], s = 2, c = 'g', label = 'Component 2')
plt.scatter(sample3[:, 0], sample3[:, 1], s = 2, c = 'b', label = 'Component 3')
plt.legend(loc='lower left')
# Q1:
print('--------------------------- K-means ---------------------------')
plt.figure(figsize = (20, 10))
K = [2, 3, 4, 5]
pos = 1
for k in K:
	# kmeans_model = KMeans(n_clusters = k, init='random', n_init=10)
	# kmeans_model.fit(sample)
	# print(kmeans_model.labels_)
	# prob_az = np.zeros((3, k))
	# pred = kmeans_model.labels_

	# for i in range(N):
	# 	prob_az[ground_truth[i], pred[i]] += 1
	
	# # P(a|z) = P(a,z) / P(z)
	# prob_az = prob_az / N
	# empirical_prob = prob_az / prob_z
	# print('For K =', k)
	# print(empirical_prob)
	# # display boundary and cluster center
	# plt.subplot(2,2,pos)
	# plt.scatter(sample[:, 0], sample[:, 1], c = pred,  s = 2)
	# plt.title('K =' + str(k))
	# # plot cluster center
	# centroids = kmeans_model.cluster_centers_
	# plt.scatter(centroids[:, 0], centroids[:, 1],
	#             marker='x', s=169, linewidths=3,
	#             color='black', zorder=10)
	# plot_boundary(sample, kmeans_model, k)

	centroids = rand_center(sample, k)
	labels, centroids = iterate_k_means(sample, centroids, 500)
	prob_az = np.zeros((3, k))
	pred = np.array(labels)

	for i in range(N):
		prob_az[ground_truth[i], pred[i]] += 1
	
	# P(a|z) = P(a,z) / P(z)
	prob_az = prob_az / N
	empirical_prob = prob_az / prob_z
	print('For K =', k)
	print(empirical_prob)
	# display boundary and cluster center
	plt.subplot(2,2,pos)
	plt.scatter(sample[:, 0], sample[:, 1], c = pred,  s = 2)
	plt.title('K =' + str(k))
	# plot cluster center
	plt.scatter(centroids[:, 0], centroids[:, 1],
	            marker='x', s=169, linewidths=3,
	            color='black', zorder=10)
	# plot_boundary(sample, kmeans_model, k)

	pos += 1
plt.suptitle('K-means')

# Q2:
print('--------------------------- EM Algorithm ---------------------------')
plt.figure(figsize = (20, 10))
pos = 1
# GMM_means = []
# GMM_cov = []
for k in K:
	GMM_model = GaussianMixture(n_components = k)
	GMM_model.fit(sample)
	
	# print(kmeans_model.labels_)
	prob_az = np.zeros((3, k))
	pred = GMM_model.predict(sample)

	for i in range(N):
		prob_az[ground_truth[i], pred[i]] += 1
	# P(a|z) = P(a,z) / P(z)
	prob_az = prob_az / N
	empirical_prob = prob_az / prob_z
	print('For K =', k) 
	print(empirical_prob)
	# display boundary and cluster center
	plt.subplot(2,2,pos)
	plt.scatter(sample[:, 0], sample[:, 1], c = pred,  s = 2)
	plt.title('K =' + str(k))
	centroids = GMM_model.means_
	plt.scatter(centroids[:, 0], centroids[:, 1],
	            marker='x', s=169, linewidths=3,
	            color='black', zorder=10)
	plot_boundary(sample, GMM_model, k)
	pos += 1

plt.suptitle('Gaussian Mixture')

# Q4
print('--------------------------- High Dimension ---------------------------')
distr = [0, -1, 1]
prob = [2/3, 1/6, 1/6]
dimen = 30
u = 7
U = generate_ortho_vec(prob, u, dimen, distr)
print('vector:\n', U)

# Q5
sigma = 0.01
# For sample1:
Z1 = np.random.normal(0, 1, round(N * pi[0]))
Z2 = np.random.normal(0, 1, round(N * pi[0]))
Noise = np.random.normal(0, sigma, size = [round(N * pi[0]), dimen])
sample1 = U[0,:] + np.outer(Z1, U[1,:]) + np.outer(Z2, U[2,:]) + Noise
label1 = np.empty(sample1.shape[0])
label1.fill(0)

# For sample2:
Z1 = np.random.normal(0, 1, round(N * pi[1]))
Z2 = np.random.normal(0, 1, round(N * pi[1]))
Noise = np.random.normal(0, sigma, size = [round(N * pi[1]), dimen])
sample2 = 2 * U[3,:] + np.sqrt(2) * np.outer(Z1, U[4,:]) + np.outer(Z2, U[5,:]) + Noise
label2 = np.empty(sample2.shape[0])
label2.fill(1)

# For sample3:
Z1 = np.random.normal(0, 1, round(N * pi[2]))
Z2 = np.random.normal(0, 1, round(N * pi[2]))
Noise = np.random.normal(0, sigma, size = [round(N * pi[2]), dimen])
sample3 = np.sqrt(2) * U[5,:] + np.outer(Z1, (U[0,:] + U[1,:])) + (1 / np.sqrt(2)) * np.outer(Z2, U[5,:]) + Noise
label3 = np.empty(sample3.shape[0])
label3.fill(2)

sample = np.concatenate((sample1, sample2, sample3), axis = 0)
label = np.concatenate((label1, label2, label3), axis = 0)
prob_z = np.array([len(label1), len(label2), len(label3)]).reshape(3, 1) / N
sample, z = reformat_data(sample, label)
ground_truth = np.argmax(z, axis = 1)

# Q6
print('--------------------------- For d = 30, K-means ---------------------------')
K = [2, 3, 4, 5]
for k in K:
	kmeans_model = KMeans(n_clusters = k, init='random', n_init=10)
	kmeans_model.fit(sample)
	
	# print(kmeans_model.labels_)
	prob_az = np.zeros((3, k))
	pred = kmeans_model.labels_

	for i in range(N):
		prob_az[ground_truth[i], pred[i]] += 1

	# P(a|z) = P(a,z) / P(z)
	prob_az = prob_az / N
	empirical_prob = prob_az / prob_z
	print('For K =', k)
	print(empirical_prob)

print('--------------------------- For d = 30, EM Algorithm ---------------------------')
for k in K:
	GMM_model = GaussianMixture(n_components = k)
	GMM_model.fit(sample)
	
	# print(kmeans_model.labels_)
	prob_az = np.zeros((3, k))
	pred = GMM_model.predict(sample)

	for i in range(N):
		prob_az[ground_truth[i], pred[i]] += 1
	# P(a|z) = P(a,z) / P(z)
	prob_az = prob_az / N
	empirical_prob = prob_az / prob_z
	print('For K =', k) 
	print(empirical_prob)

# plt.show()
