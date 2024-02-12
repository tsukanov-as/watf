import numpy as np

from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.05, random_state=42) # type: ignore

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