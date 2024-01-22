import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1000, random_state=42, train_size=6000)
x_train = x_train.values.astype(int)
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(int)
y_test = y_test.values.astype(int)

input_count = 28*28
hidden_count = 2000
output_count = 10

import numpy as np
kernels = np.random.choice([-1, 1], size=(hidden_count, input_count), p=[1./2, 1./2])

import watf

c = watf.Сlassifier(output_count, hidden_count)

# precalculation of all at once (requires a lot of memory)
hidden_train = np.sign(np.einsum('ik,jk->ij', x_train, kernels)).clip(0)
hidden_test = np.sign(np.einsum('ik,jk->ij', x_test, kernels)).clip(0)

# train
for epoch in range(50):
    total_misses = c.tune_all(y_train, hidden_train)

    train_accuracy = c.test_all(y_train, hidden_train)
    test_accuracy = c.test_all(y_test, hidden_test)
    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, train_accuracy, test_accuracy))

    if total_misses == 0:
        break
