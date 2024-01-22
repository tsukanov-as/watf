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

import numpy as np

W = (np.random.rand(shard_count, input_count))
T = np.full(shard_count, 10.0)

shards = np.zeros(len(x_train), dtype=np.int64)

import random

# train shards
for epoch in range(50):
    w = (W.T / T).T
    stat = np.zeros(shard_count)
    for i in range(len(x_train)):
        fv = x_train[i]
        res = (np.sum((w-fv)**2, axis=1))
        win = res.argmin()
        shards[i] = win
        stat[win] += 1
        W[win] += fv
        T[win] += 1
    print(epoch, len(stat[stat == 0]), stat)
    for j, v in enumerate(stat):
        if v < 2:
            W[j] = x_train[random.randint(0, len(x_train)-1)]
            T[j] = 1

import watf

cc = []
for i in range(shard_count):
    cc.append(watf.Сlassifier(output_count, input_count))

def shard(fv):
    return np.sum((w-fv)**2, axis=1).argmin()

# train
for epoch in range(200):
    total_misses = 0
    for i in range(len(x_train)):
        c = cc[shards[i]]
        if c.tune(y_train[i], x_train[i]):
            total_misses += 1

    total_test = 0
    for i in range(len(x_test)):
        c = cc[shard(x_test[i])]
        if y_test[i] == c.pred(x_test[i]):
            total_test += 1

    total_train = 0
    for i in range(len(x_train)):
        c = cc[shards[i]]
        if y_train[i] == c.pred(x_train[i]):
            total_train += 1

    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, total_train/len(x_train), total_test/len(x_test)))

    if total_misses == 0:
        break