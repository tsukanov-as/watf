from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.05, random_state=42) # type: ignore

input_count = 8*8
hidden_count = 1000
output_count = 10

import numpy as np
kernels = np.random.choice([-1, 1], size=(hidden_count, input_count), p=[1./2, 1./2])

import watf

c = watf.Сlassifier(output_count, hidden_count)

# precalculation of all at once (requires a lot of memory)
hidden_train = np.heaviside(np.einsum('ik,jk->ij', x_train, kernels), 0)
hidden_test = np.heaviside(np.einsum('ik,jk->ij', x_test, kernels), 0)

# train
for epoch in range(50):
    total_misses = c.tune_all(y_train, hidden_train)

    train_accuracy = c.test_all(y_train, hidden_train)
    test_accuracy = c.test_all(y_test, hidden_test)
    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, train_accuracy, test_accuracy))

    if total_misses == 0:
        break
