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

input_count = 26*26*8
output_count = 10

kernels = [
    [[
        [  0,-1,-1],
        [ +1, 0,-1],
        [ +1,+1, 0],
    ]],
    [[
        [  0,+1,+1],
        [ -1, 0,+1],
        [ -1,-1, 0],
    ]],
    [[
        [ -1,-1, 0],
        [ -1, 0,+1],
        [  0,+1,+1],
    ]],
    [[
        [ +1,+1, 0],
        [ +1, 0,-1],
        [  0,-1,-1],
    ]],
    [[
        [ +1,+1,+1],
        [  0, 0, 0],
        [ -1,-1,-1],
    ]],
    [[
        [ -1,-1,-1],
        [  0, 0, 0],
        [ +1,+1,+1],
    ]],
    [[
        [ +1, 0,-1],
        [ +1, 0,-1],
        [ +1, 0,-1],
    ]],
    [[
        [ -1, 0,+1],
        [ -1, 0,+1],
        [ -1, 0,+1],
    ]],
]

kernels = torch.tensor(kernels ,dtype=torch.float64)

import watf

c = watf.Сlassifier(output_count, input_count)

def hidden(x):
    """ fixed hidden layer """
    return conv2d(x, kernels, padding=0).reshape(input_count).clip(0).numpy()

# train
for epoch in range(20):
    total_misses = 0
    for i in range(len(x_train)):
        if c.tune(y_train[i], hidden(x_train[i])):
            total_misses += 1

    total_test = 0
    for i in range(len(x_test)):
        if y_test[i] == c.pred(hidden(x_test[i])):
            total_test += 1

    total_train = 0
    for i in range(len(x_train)):
        if y_train[i] == c.pred(hidden(x_train[i])):
            total_train += 1

    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, total_train/len(x_train), total_test/len(x_test)))

    if total_misses == 0:
        break