import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1000, random_state=42, train_size=6000)
x_train = x_train.values.astype(int)
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(int)
y_test = y_test.values.astype(int)

input_count = 28*28
hidden_count = 1000
output_count = 10

import numpy as np
kernels = np.random.choice([-1, 1], size=(hidden_count, input_count), p=[1./2, 1./2])

from sklearn.linear_model import LogisticRegression

c = LogisticRegression(penalty=None, solver="saga")

# precalculation of all at once (requires a lot of memory)
hidden_train = np.einsum('ik,jk->ij', x_train, kernels).clip(0)
hidden_test = np.einsum('ik,jk->ij', x_test, kernels).clip(0)

c.fit(hidden_train, y_train)

total_test = 0
for i in range(len(x_test)):
    if y_test[i] == c.predict([hidden_test[i]])[0]:
        total_test += 1

total_train = 0
for i in range(len(x_train)):
    if y_train[i] == c.predict([hidden_train[i]])[0]:
        total_train += 1

print("accuracy train: %f; accuracy test: %f" % (total_train/len(x_train), total_test/len(x_test)))