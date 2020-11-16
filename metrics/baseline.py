# Dirty hack to have baseline in subfolder run from root with python3 metrics/baseline.py
from os.path import dirname as dir
from sys import path

path.append(dir(path[0]))

from peer.model.femnist import ClientModel
from peer.model.utils.model_utils import read_data
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

c_model = ClientModel(0)

train_clients, train_groups, train_data, test_data = read_data('data/train', 'data/test')

x_train = []
y_train = []

print(len(train_clients))
for c in train_clients:
    x_train.extend(train_data[c]["x"])
    y_train.extend(train_data[c]["y"])

train_data = {"x": x_train, "y": y_train}

x_test = []
y_test = []

for c in train_clients:
    x_test.extend(test_data[c]["x"])
    y_test.extend(test_data[c]["y"])

print(len(x_train))

test_data = {"x": x_test, "y": y_test}

print(len(x_test))

c_model.train(train_data, num_epochs=30)

res = c_model.test(test_data)
print("Accuracy:", res["accuracy"])
print("Loss:", res["loss"])
