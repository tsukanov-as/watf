import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

import numpy as np

x_train = np.reshape(x_train, (50000, 32*32*3)).astype(int)
x_test = np.reshape(x_test, (10000, 32*32*3)).astype(int)

y_train = y_train.ravel()
y_test = y_test.ravel()

from sklearn.linear_model import Perceptron
c = Perceptron(tol=1e-6, penalty=None, verbose=1)
c.fit(x_train, y_train)

total_test = 0
for i in range(len(x_test)):
    if y_test[i] == c.predict([x_test[i]])[0]:
        total_test += 1

total_train = 0
for i in range(len(x_train)):
    if y_train[i] == c.predict([x_train[i]])[0]:
        total_train += 1

print("accuracy train: %f; accuracy test: %f" % (total_train/len(x_train), total_test/len(x_test)))