from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.05, random_state=42) # type: ignore

input_count = 8*8
hidden_count = 2000
output_count = 10

import numpy as np
kernels = np.random.choice([-1, 0, 1], size=(hidden_count, input_count), p=[1./3, 1./3, 1./3])

import watf

c = watf.Сlassifier(output_count, hidden_count)

def hidden(x):
    """ fixed hidden layer """
    return np.sign(kernels @ x).clip(0)

# train
for i in range(len(x_train)):
    c.feed(y_train[i], hidden(x_train[i]))

# test
total = 0
for i in range(len(x_test)):
    if y_test[i] == c.pred(hidden(x_test[i])):
        total += 1
print("accuracy:", total/len(x_test))

# tune
for epoch in range(100):
    for i in range(len(x_train)):
        c.tune(y_train[i], hidden(x_train[i]))

    total_test = 0
    for i in range(len(x_test)):
        if y_test[i] == c.pred(hidden(x_test[i])):
            total_test += 1

    total = 0
    for i in range(len(x_train)):
        if y_train[i] == c.pred(hidden(x_train[i])):
            total += 1
    print("accuracy train/test:", total/len(x_train), total_test/len(x_test))
