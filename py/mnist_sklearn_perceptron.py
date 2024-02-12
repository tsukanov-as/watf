import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42, train_size=60000)
x_train = x_train.values.astype(int)
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(int)
y_test = y_test.values.astype(int)

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