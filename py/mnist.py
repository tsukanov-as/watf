import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42, train_size=60000)
x_train = x_train.values.astype(int)
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(int)
y_test = y_test.values.astype(int)

input_count = 28*28
output_count = 10

import watf

c = watf.Сlassifier(output_count, input_count)

# train
for i in range(len(x_train)):
    c.feed(y_train[i], x_train[i])

# test
total = 0
for i in range(len(x_test)):
    if y_test[i] == c.watf(x_test[i]).argmax():
        total += 1

print("accuracy:", total/len(x_test))