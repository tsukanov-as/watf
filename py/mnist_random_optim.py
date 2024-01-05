import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1000, random_state=42, train_size=3000)
x_train = x_train.values.astype(int)
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(int)
y_test = y_test.values.astype(int)

input_count = 28*28
hidden_count = 3000
output_count = 10

import numpy as np
kernels = np.random.choice([-1, 0, 1], size=(hidden_count, input_count), p=[1./3, 1./3, 1./3])

import watf

c = watf.Сlassifier(output_count, hidden_count)

# precalculation of all at once (requires a lot of memory)
hidden_train = np.sign(np.einsum('ik,jk->ij', x_train, kernels)).clip(0)
hidden_test = np.sign(np.einsum('ik,jk->ij', x_test, kernels)).clip(0)

# train
for i in range(len(hidden_train)):
    c.feed(y_train[i], hidden_train[i])

# test
total = 0
for i in range(len(hidden_test)):
    if y_test[i] == c.pred(hidden_test[i]):
        total += 1
print("accuracy:", total/len(hidden_test))

# tune
for epoch in range(200):
    for i in range(len(hidden_train)):
        c.tune(y_train[i], hidden_train[i])

    total_test = 0
    for i in range(len(x_test)):
        if y_test[i] == c.pred(hidden_test[i]):
            total_test += 1

    total = 0
    for i in range(len(x_train)):
        if y_train[i] == c.pred(hidden_train[i]):
            total += 1
    print("accuracy train/test:", total/len(x_train), total_test/len(x_test))