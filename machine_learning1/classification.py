"""
Classification problem with kNN, LinearClassifier
"""

import argparse
import _pickle as _pic
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


def read_data(names):
    X = []
    y = []
    for name in names:
        with open(name, "rb") as f:
            data = _pic.load(f, encoding="bytes")
        X.append(data[b"data"])
        y.append(data[b"labels"])
    # merge arrays
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y

data_names = ["mnist_dataset\data_batch_1", "mnist_dataset\data_batch_3", "mnist_dataset\data_batch_2",
              "mnist_dataset\data_batch_4", "mnist_dataset\data_batch_5"]

test_names = ["mnist_dataset\test_batch"]


def train_and_test(train_X, train_y, test_X, test_y, m="LC", par=1):
    if m == "kNN":
        clf = KNeighborsClassifier(n_neighbors=par)
    else:
        clf = LogisticRegression(max_iter=par)
    print("Training")
    start = time.time()
    clf.fit(train_X, train_y)
    print("Elapsed: %f" % (time.time() - start))
    print("Testing")
    start = time.time()
    test = clf.score(test_X, test_y)
    print("test_A = %f" % test)
    print("Elapsed: %f" % (time.time() - start))
    return test


num_arr = len(data_names)
num_iter = 5

result = np.zeros((num_iter, num_arr))
for param in range(1, 2 * num_iter, 2):
    for i in range(num_arr):
        data_train = data_names[:i] + data_names[i+1:]
        data_test = [data_names[i]]
        train_X, train_y = read_data(data_train)
        test_X, test_y = read_data(data_test)
        train_y = train_y.flatten()
        test_y = test_y.flatten()

        train_ind = np.arange(train_X.shape[0])
        np.random.shuffle(train_ind)
        train_ind = train_ind[:10000]

        test_ind = np.arange(test_X.shape[0])
        np.random.shuffle(test_ind)
        test_ind = test_ind[:10000]

        train_X, train_y = train_X[train_ind], train_y[train_ind]
        test_X, test_y = test_X[test_ind], test_y[test_ind]

        print("Validation N:%i" % param)
        result[param - 1, i] = train_and_test(train_X, train_y, test_X, test_y, m="kNN", par=param)

print(result)
m1 = np.mean(result, axis=1)
sq_sum = np.sum(np.power(result, 2), axis=1)
sum_sq = np.power(np.sum(result, axis=1), 2)
var = np.sqrt((sq_sum - sum_sq / num_arr)) / (num_arr - 1)
#var = np.var(result, axis=1)
print(m1)
print(sq_sum)
print(sum_sq)
print(var)


for x, y in zip(range(1, num_iter + 1), result[:]):
    plt.scatter([x] * num_arr, y)
    variances = [m1[x-1] - var[x-1], m1[x-1] + var[x-1]]
    plt.scatter([x] * 2, variances, marker="_", c='black')
    plt.plot(x, m1[x-1], marker='1', c="black")
plt.savefig("Result_fig.png")
plt.show()


# check guts
# if args.clf == "LC":
#     for l, w in enumerate(clf.coef_):
#         plt.figure()
#         plt.title(l)
#         plt.imshow(w.reshape((28, 28)))
#         plt.gray()
#     plt.show()