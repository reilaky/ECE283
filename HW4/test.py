import numpy as np

test = np.random.rand(10, 2)
print(test)
y = np.random.choice([0.0, 1.0], 10, p = [0.4, 0.6])
print(y)
pred = np.ones(np.shape(y))
pred[test[:, 0] < .5] = 0

print(test[:, 0] < .5)
print(pred)

w = np.full(np.shape(y), 0.1)
error = sum(w[y != pred])
print(error)
print(y != pred)
print(w[y != pred])