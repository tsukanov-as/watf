import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

import numpy as np

x_train = np.reshape(x_train, (50000, 32*32*3))
x_test = np.reshape(x_test, (10000, 32*32*3))

y_train = y_train.ravel()
y_test = y_test.ravel()

input_count = 32*32*3
output_count = 10

import watf

c = watf.Сlassifier(output_count, input_count)

# train
for i in range(len(x_train)):
    c.feed(y_train[i], x_train[i])

# test
total = 0
for i in range(len(x_test)):
    if y_test[i] == c.pred(x_test[i]):
        total += 1
print("accuracy:", total/len(x_test))

# tune
for epoch in range(20):
    for i in range(len(x_train)):
        c.tune(y_train[i], x_train[i])

    total = 0
    for i in range(len(x_test)):
        if y_test[i] == c.pred(x_test[i]):
            total += 1
    print("accuracy:", total/len(x_test))