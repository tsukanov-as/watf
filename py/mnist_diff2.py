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

import diff2

c = diff2.Сlassifier(output_count, input_count)

# train
for epoch in range(50):
    total_misses = c.tune_all(y_train, x_train)

    train_accuracy = c.test_all(y_train, x_train)
    test_accuracy = c.test_all(y_test, x_test)
    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, train_accuracy, test_accuracy))

    if total_misses == 0:
        break

# import matplotlib.pyplot as plt
# for i, w in enumerate(c.W):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(w.reshape((28,28)), cmap=plt.get_cmap('gray'), interpolation="nearest")
#     plt.xticks(())
#     plt.yticks(())
# plt.show()