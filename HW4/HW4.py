
import numpy as np
import matplotlib.pyplot as plt
from Adaboost import Adaboost
from sklearn import svm

N = 400
N_TEST = 100

def generateGuassian(mean, theta, lambd, num):
    u1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    u2 = np.array([[-np.sin(theta)], [np.cos(theta)]])
    cov = lambd[0] * np.dot(u1, np.transpose(u1)) + lambd[1] * np.dot(u2, np.transpose(u2))
    sample = np.random.multivariate_normal(mean, cov, num)
    return sample, cov

def randomize(dataset, label):
    permutation = np.random.permutation(label.shape[0])
    shuffled_training_dataset = dataset[permutation, :]
    shuffled_label = label[permutation]
    return [shuffled_training_dataset, shuffled_label]


def plot_dataset(dataset, label):
    plt.scatter(dataset[:, 0], dataset[:, 1], s = 2, c = label)

def plot_decisionboundary(model, X, pred, label, clfs = None):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    if clfs == None:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()], clfs)
    

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot dataset
    plt.scatter(X[:, 0], X[:, 1], s = 2, c = label)

    # Plot incorrect prediction
    indexs =  np.asarray(np.where((pred != test_label) == True))
    print('Incorrect prob:', float(indexs.shape[1] / X.shape[0]) * 100, '%')
    plt.scatter(X[indexs, 0], X[indexs, 1], s=16, facecolors='none', edgecolors='k', label = 'Incorrect prediction')
    plt.legend(loc = 'upper left')
# Define kernels
def gaussian_kernel(X1, X2, sigma=0.1):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(-np.sum(np.power((x1 - x2), 2)) / float(2*(sigma**2)))
    return gram_matrix



n = N // 2
n_test = N_TEST // 2
# for Class0
mean = [0, 0]
theta = 0
lambd = [2, 1]
sample0, cov = generateGuassian(mean, theta, lambd, n)
test0, cov = generateGuassian(mean, theta, lambd, n_test)
print('Class0 cov = \n', cov)
label0 = np.full(n, 0)

# for Class1
meanA = [-4, 2]
piA = 1 / 3
thetaA = - np.pi * 3 / 4
lambdaA = [2, 1/4]
sample1A, covA = generateGuassian(meanA, thetaA, lambdaA, round(n * piA))
test1A, covA = generateGuassian(meanA, thetaA, lambdaA, round(n_test * piA))
print('Class1 covA = \n', covA)

meanB = [6, 4]
piB = 2 / 3
thetaB = np.pi / 4
lambdaB = [3, 1]
sample1B, covB = generateGuassian(meanB, thetaB, lambdaB, round(n * piB))
test1B, covB = generateGuassian(meanB, thetaB, lambdaB, round(n_test * piB))
print('Class1 covB = \n', covB)
sample1 = np.concatenate((sample1A, sample1B), axis = 0)
test1 = np.concatenate((test1A, test1B), axis = 0)
label1 = np.full(n, 1)

raw_dataset = np.concatenate((sample0, sample1), axis = 0)
raw_label = np.concatenate((label0, label1), axis = 0)

dataset, label = randomize(raw_dataset, raw_label)

testset = np.concatenate((test0, test1), axis = 0)
test_label = np.concatenate((np.full(n_test, 0), np.full(n_test, 1)), axis = 0)


# SVM
plt.figure(figsize = (20, 30))
plt.subplot(2, 2, 1)
plt.title('Training set')
plot_dataset(dataset, label)
print('---------------------------------- SVM ----------------------------------')
sigma = 0.3
gamma = 1.0 / (2 * (sigma ** 2))
# rbf: exp(-gamma *||x-x'||^2). gamma is specified by keyword gamma, must be greater than 0.
clf = svm.SVC(kernel='rbf', gamma = gamma)
clf.fit(dataset, label)
# (test, dataset)
support_vectors = clf.support_vectors_
print('Support vectors:')
print(support_vectors)
print('Size of support vectors:', np.shape(support_vectors))
print('Fraction of support vectors', float(support_vectors.shape[0] / N) * 100, '%')
# for trainingset

print('\nFor training set:')
plt.subplot(2, 2, 2)
plt.title('SVM')
plot_dataset(dataset, label)

pred = clf.predict(dataset)
plot_decisionboundary(clf, dataset, pred, label)
# plot support vector
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s = 16, c = 'black', marker = '+', label = 'Support vectors')
plt.legend(loc = 'upper left')

# for testset
print('\nFor testset: ')
plt.subplot(2, 2, 3)
plt.title('Testset')
plot_dataset(testset, test_label)

plt.subplot(2, 2, 4)
plt.title('SVM')
pred = clf.predict(testset)
plot_decisionboundary(clf, testset, pred, test_label)


# Adaboost
print('---------------------------------- Adaboost ----------------------------------')
plt.figure(figsize = (20, 30))

plt.subplot(1, 2, 1)
plt.title('Training set')
plot_dataset(dataset, label)

clf = Adaboost(iters = 5000, thres = 0.05)
clf.fit(dataset, label)
# for training set
pred = clf.predict(dataset)

plt.subplot(1, 2, 2)
plt.title('Adaboost')
plot_decisionboundary(clf, dataset, pred, label)

num = 5
if clf.M < num:
    num = clf.M
print('M =', clf.M)
print('first five weak learners:')
print('feature\tthreshold')
for i in range(num):
    print(clf.clfs[i].feature_index, '\t', clf.clfs[i].threshold)

# for test set
plt.figure(figsize = (20, 30))
plt.title('Testset')
first_n_weak_learners = []
for i in range(num):
    title = 'No.' + str(i + 1) + ' weak learner'
    plt.subplot(2, 3, i+1)
    plt.title(title)
    print('For', title + ':')
    
    pred = clf.predict(testset, [clf.clfs[i]])
    plot_decisionboundary(clf, testset, pred, test_label, [clf.clfs[i]])
    first_n_weak_learners.append(clf.clfs[i])

plt.figure(figsize = (20, 30))
plt.subplot(1, 2, 1)
title = 'First five weak learners'
plt.title(title)
print('For', title + ':')
pred = clf.predict(testset, first_n_weak_learners)
plot_decisionboundary(clf, testset, pred, test_label, first_n_weak_learners)

title = 'All weak learners'
plt.subplot(1, 2, 2)
plt.title(title)
print('For', title + ':')
pred = clf.predict(testset)
plot_decisionboundary(clf, testset, pred, test_label)


plt.show()
