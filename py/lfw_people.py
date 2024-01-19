import numpy as np

from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=100)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(people.data, people.target, test_size=0.05, random_state=42) # type: ignore

input_count = 62*47
output_count = people.target.shape[0]

import watf3 as watf

c = watf.Сlassifier(output_count, input_count)

# train
for epoch in range(100):
    total_misses = c.tune_all(y_train, x_train)

    train_accuracy = c.test_all(y_train, x_train)
    test_accuracy = c.test_all(y_test, x_test)
    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, train_accuracy, test_accuracy))

    if total_misses == 0:
        break