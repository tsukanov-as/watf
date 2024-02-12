import time
import numpy as np

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42, train_size=60000)
x_train = x_train.values.astype(float)
y_train = y_train.values.astype(int)
x_test = x_test.values.astype(float)
y_test = y_test.values.astype(int)

from sklearn.cluster import KMeans

clusters_sqrt = 6
clusters = clusters_sqrt**2

rng = np.random.RandomState(0)
kmeans = KMeans(init="k-means++", n_clusters=clusters, random_state=rng, verbose=True, n_init=1, max_iter=50)

t0 = time.time()
kmeans.fit(x_train)
dt = time.time() - t0
print("done in %.2fs." % dt)

shard_count = clusters
shards_train = kmeans.predict(x_train)
shards_test = kmeans.predict(x_test)

from sklearn.linear_model import Perceptron

cc = []
for i in range(shard_count):
    c = Perceptron(tol=1e-3, penalty=None)
    c.fit(x_train[shards_train == i], y_train[shards_train == i])
    cc.append(c)

total_test = 0
for i in range(len(x_test)):
    c = cc[shards_test[i]]
    if y_test[i] == c.predict([x_test[i]])[0]:
        total_test += 1

total_train = 0
for i in range(len(x_train)):
    c = cc[shards_train[i]]
    if y_train[i] == c.predict([x_train[i]])[0]:
        total_train += 1

print("accuracy train: %f; accuracy test: %f" % (total_train/len(x_train), total_test/len(x_test)))

# import matplotlib.pyplot as plt

# for i, patch in enumerate(kmeans.cluster_centers_):
#     plt.subplot(clusters_sqrt, clusters_sqrt, i + 1)
#     plt.imshow(patch.reshape((28,28)), cmap=plt.cm.gray, interpolation="nearest")
#     plt.xticks(())
#     plt.yticks(())
# plt.show()