import time
import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1000, random_state=42, train_size=1000)
x_train = x_train.values.astype(int)
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(int)
y_test = y_test.values.astype(int)

images = x_train.reshape((-1, 28, 28))

from sklearn.feature_extraction.image import extract_patches_2d

patch_size = (5, 5)
input_count = np.prod(patch_size)

clusters_sqrt = 5
clusters = clusters_sqrt**2

W = np.zeros((clusters, input_count), dtype=int)
T = np.ones((clusters, 1), dtype=int)
w = np.zeros((clusters, input_count), dtype=int)

buffer = []
start = time.time()

epochs = 5
index = 0
for epoch in range(epochs):
    i = 0
    for img in images:
        patches_2d = extract_patches_2d(img, patch_size)
        patches = np.reshape(patches_2d, (len(patches_2d), -1))
        for path in patches:
            win = np.sum((w-path)**2, axis=1).argmin()
            W[win] += path
            T[win] += 1
        i += 1
        if i % 50 == 0:
            w = W // T
            W.fill(0)
            T.fill(1)
    print(epoch)

dt = time.time() - start
print("done in %.2fs." % dt)

import matplotlib.pyplot as plt

for i, patch in enumerate(w):
    plt.subplot(clusters_sqrt, clusters_sqrt, i + 1)
    plt.imshow(patch.reshape(patch_size), cmap=plt.cm.gray, interpolation="nearest", vmin=0, vmax=255)
    plt.xticks(())
    plt.yticks(())

plt.show()