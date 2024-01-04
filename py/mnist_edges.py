import torch
from torch.nn.functional import conv2d

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42, train_size=60000)
x_train = torch.from_numpy(x_train.values.astype(float)).reshape((-1, 1, 28, 28))
y_train = torch.from_numpy(y_train.values.astype(int))
x_test  = torch.from_numpy(x_test.values.astype(float)).reshape((-1, 1, 28, 28))
y_test  = torch.from_numpy(y_test.values.astype(int))

input_count = 27*27*6
output_count = 10

kernels = [
    [[
        [ +1,-1],
        [ -1,+1],
    ]],
    [[
        [ -1,+1],
        [ +1,-1],
    ]],
    [[
        [ +1,+1],
        [ -1,-1],
    ]],
    [[
        [ -1,-1],
        [ +1,+1],
    ]],
    [[
        [ +1,-1],
        [ +1,-1],
    ]],
    [[
        [ -1,+1],
        [ -1,+1],
    ]],
]

kernels = torch.tensor(kernels ,dtype=torch.float64)

import watf

c = watf.Сlassifier(output_count, input_count)
c.W = torch.from_numpy(c.W)

def edges(x):
    """ fixed hidden layer """
    return conv2d(x, kernels, padding=0).reshape(input_count).clip(0)

# train
for i in range(len(x_train)):
    c.feed(y_train[i], edges(x_train[i]))

# test
total = 0
for i in range(len(x_test)):
    if y_test[i] == c.pred(edges(x_test[i])):
        total += 1
print("accuracy:", total/len(x_test))

# tune
for epoch in range(20):
    for i in range(len(x_train)):
        c.tune(y_train[i], edges(x_train[i]))

    total = 0
    for i in range(len(x_test)):
        if y_test[i] == c.pred(edges(x_test[i])):
            total += 1
    print("accuracy:", total/len(x_test))