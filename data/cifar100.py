from functools import partial
import math

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.python.data import AUTOTUNE

from data.abstract_dataset import AbstractDataset


class CIFAR100Dataset(AbstractDataset):
    @property
    def name(self):
        return 'CIFAR100'

    @property
    def class_labels(self):
        return None

    @property
    def dataset_size(self):
        # https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data
        return {'train': 50000, 'test': 10000}

    @property
    def avg_local_dataset_size(self):
        return 100.0, 0.0

    def get_default_accuracy(self):
        return 0.553

    def get_dataset_size_for_client(self, client_id):
        return self.avg_local_dataset_size[0]

    def _load_tff_dataset(self, distribution='equal'):
        cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
        label_key = 'coarse_label' if self.cfg.num_classes == 20 else 'label'

        def element_fn(element):
            if self.normalisation_mean_zero:
                return element['image'] / 127.5 - 1, element[label_key]
            else:
                return element['image'] / 255, element[label_key]

        def preprocess_federated_dataset(dataset, buffer_size, cache=True):
            preprocessed_ds = (dataset.shuffle(buffer_size=buffer_size,
                                               reshuffle_each_iteration=False)
                                      .map(element_fn, num_parallel_calls=AUTOTUNE))
            if cache:
                return preprocessed_ds.cache()
            return preprocessed_ds

        complete_train_ds = cifar_train.preprocess(partial(preprocess_federated_dataset,
                                                           buffer_size=100,
                                                           cache=False))
        self.train_ds = complete_train_ds.preprocess(
            lambda ds: ds.take(math.floor(100 * (1 - self.cfg.val_fraction))).cache()
        )
        self.val_ds = complete_train_ds.preprocess(
            lambda ds: ds.skip(math.floor(100 * (1 - self.cfg.val_fraction))).cache()
        )

        # need to redistribute test data, since it is made for 100 clients
        preprocessed_test_ds = cifar_test.preprocess(partial(preprocess_federated_dataset,
                                                     buffer_size=100,
                                                     cache=False))

        def split_test_ds_of_client(client_id):
            return [list(preprocessed_test_ds.create_tf_dataset_for_client(client_id).skip(20*i).take(20).as_numpy_iterator()) for i in range(5)]

        new_distributed_test_data = [split_test_ds_of_client(client_id) for client_id in preprocessed_test_ds.client_ids]
        new_distributed_test_data = [ds for client_ds in new_distributed_test_data for ds in client_ds]

        def get_new_client_data(client_id):
            single_client_data = new_distributed_test_data[int(client_id)]
            x = [data[0] for data in single_client_data]
            y = [data[1] for data in single_client_data]
            return tf.data.Dataset.from_tensor_slices((x, y))

        self.test_ds = tff.simulation.datasets.ClientData.from_clients_and_fn([str(i) for i in range(500)],
                                                                              get_new_client_data)
