import time
import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42, train_size=60000)
x_train = x_train.values.astype(int)
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(int)
y_test = y_test.values.astype(int)

from sklearn.cluster import KMeans

clusters_sqrt = 30
clusters = clusters_sqrt**2

rng = np.random.RandomState(0)
kmeans = KMeans(init="k-means++", n_clusters=clusters, random_state=rng, verbose=True, n_init=1, max_iter=100)

t0 = time.time()
kmeans.fit(x_train)
dt = time.time() - t0
print("done in %.2fs." % dt)

input_count = 28*28
output_count = 10

import watf

c = watf.Сlassifier(output_count, clusters)

w = kmeans.cluster_centers_

r = 256*3

def gaussian(alpha):
    return np.exp(-(alpha**2)/(r**2))

def rbf(x):
    return gaussian(np.sqrt(np.sum((w-x)**2, axis=1)))

clusters_train = np.zeros((len(x_train), clusters), dtype=float)
for i in range(len(x_train)):
    clusters_train[i] = rbf(x_train[i])

clusters_test = np.zeros((len(x_test), clusters), dtype=float)
for i in range(len(x_test)):
    clusters_test[i] = rbf(x_test[i])

# train
for epoch in range(20):
    total_misses = 0
    
    for i in range(len(x_train)):
        if c.tune(y_train[i], clusters_train[i]):
            total_misses += 1

    total_test = 0
    for i in range(len(x_test)):
        if y_test[i] == c.pred(clusters_test[i]):
            total_test += 1

    total_train = 0
    for i in range(len(x_train)):
        if y_train[i] == c.pred(clusters_train[i]):
            total_train += 1

    print("[epoch %d] accuracy train: %f; accuracy test: %f" % (epoch, total_train/len(x_train), total_test/len(x_test)))

    if total_misses == 0:
        break


# import matplotlib.pyplot as plt

# for i, patch in enumerate(kmeans.cluster_centers_):
#     plt.subplot(clusters_sqrt, clusters_sqrt, i + 1)
#     plt.imshow(patch.reshape((28,28)), cmap=plt.cm.gray, interpolation="nearest")
#     plt.xticks(())
#     plt.yticks(())
# plt.show()