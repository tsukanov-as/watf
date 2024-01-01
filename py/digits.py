import numpy as np

from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.05, random_state=42) # type: ignore

input_count = 8*8
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