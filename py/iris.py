from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

input_count = 4
output_count = 3

import watf3 as watf

c = watf.Сlassifier(output_count, input_count)

# train
for epoch in range(50):
    total_misses = c.tune_all(y_train, x_train)

    train_accuracy = c.test_all(y_train, x_train)
    test_accuracy = c.test_all(y_test, x_test)
    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, train_accuracy, test_accuracy))

    if total_misses == 0:
        break