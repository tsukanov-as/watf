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
for epoch in range(20):
    total_misses = c.tune_all(y_train, x_train)

    train_accuracy = c.test_all(y_train, x_train)
    test_accuracy = c.test_all(y_test, x_test)
    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, train_accuracy, test_accuracy))

    if total_misses == 0:
        break
