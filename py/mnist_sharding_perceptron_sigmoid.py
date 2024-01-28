import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1000, random_state=42, train_size=10000)
x_train = x_train.values.astype(float) / 255
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(float) / 255
y_test = y_test.values.astype(int)

input_count = 28*28
shard_count = 4
output_count = 10

import numpy as np

W = np.zeros((shard_count, input_count), dtype=float)
T = np.ones((shard_count, 1), dtype=float)

shards_train = np.zeros(len(x_train), dtype=int)

import random

for i in range(shard_count):
    W[i] = x_train[random.randint(0, len(x_train)-1)]
    T[i] = 1

w = W / T

def shard(fv):
    return np.sum((w-fv)**2, axis=1).argmin()

# train shards
for epoch in range(50):
    stat = np.zeros(shard_count)
    for i in range(len(x_train)):
        win = shard(x_train[i])
        shards_train[i] = win
        stat[win] += 1
        W[win] += x_train[i]
        T[win] += 1
    w = W / T
    print(epoch, stat)

import perceptron_sigmoid as c

cc = []
for i in range(shard_count):
    cc.append(c.Сlassifier(output_count, input_count))

shards_test = np.zeros(len(x_test), dtype=int)
for i in range(len(x_test)):
    shards_test[i] = shard(x_test[i])

y_train_eye = np.eye(10)[y_train]

# train
for epoch in range(200):
    for i in range(len(x_train)):
        c = cc[shards_train[i]]
        c.tune(y_train_eye[i], x_train[i])

    total_test = 0
    for i in range(len(x_test)):
        c = cc[shards_test[i]]
        if y_test[i] == c.pred(x_test[i]):
            total_test += 1

    total_train = 0
    for i in range(len(x_train)):
        c = cc[shards_train[i]]
        if y_train[i] == c.pred(x_train[i]):
            total_train += 1

    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, total_train/len(x_train), total_test/len(x_test)))
