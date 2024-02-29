import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1000, random_state=42, train_size=10000)
x_train = x_train.values.astype(int)
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(int)
y_test = y_test.values.astype(int)

input_count = 28*28
shard_count = 4
output_count = 10

kernels = np.random.choice([-1, 1], size=(shard_count, input_count), p=[1./2, 1./2])
# kernels = np.random.rand(shard_count, input_count)

import watf

cc = []
for i in range(shard_count):
    cc.append(watf.Сlassifier(output_count, input_count))

# precalculation of all at once (requires a lot of memory)
hidden_train = np.einsum('ik,jk->ij', x_train, kernels).argmax(axis=1)
hidden_test = np.einsum('ik,jk->ij', x_test, kernels).argmax(axis=1)

# train
for epoch in range(200):
    total_misses = 0
    for i in range(len(x_train)):
        c = cc[hidden_train[i]]
        if c.tune(y_train[i], x_train[i]):
            total_misses += 1

    total_test = 0
    for i in range(len(x_test)):
        c = cc[hidden_test[i]]
        if y_test[i] == c.pred(x_test[i]):
            total_test += 1

    total_train = 0
    for i in range(len(x_train)):
        c = cc[hidden_train[i]]
        if y_train[i] == c.pred(x_train[i]):
            total_train += 1

    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, total_train/len(x_train), total_test/len(x_test)))

    if total_misses == 0:
        break

noise = np.random.choice([-1, 1], size=x_train.shape, p=[1./2, 1./2])
x_train += noise

hidden_train = np.einsum('ik,jk->ij', x_train, kernels).argmax(axis=1)

total_train = 0
for i in range(len(x_train)):
    c = cc[(hidden_train[i])]
    if y_train[i] == c.pred(x_train[i]):
        total_train += 1

print("accuracy train with noise: %f" % (total_train/len(x_train)))