from functools import partial
import logging
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow_federated.python.simulation.datasets import ClientData
from models.utils.language_utils import letter_to_index_poets, word_to_indices_poets

from data.abstract_dataset import AbstractDataset


class PoetsDataset(AbstractDataset):
    @property
    def class_labels(self):
        raise NotImplementedError

    @property
    def dataset_size(self):
        return {'train': 177457, 'test': 22831}

    @property
    def avg_local_dataset_size(self):
        assert self._local_dataset_sizes
        total_local_sizes = [np.sum(size) for size in self._local_dataset_sizes.values()]
        return np.mean(total_local_sizes), np.std(total_local_sizes)

    def get_default_accuracy(self):
        return 0.858    # todo

    def get_dataset_size_for_client(self, client_id):
        assert self._local_dataset_sizes
        assert not self.centralised
        return self._local_dataset_sizes[client_id]

    def _load_tff_dataset_from_files(self):
        logging.info('Loading data from files...')

        def create_tf_dataset_for_client_fn(data_dict, partition, client_id):
            if partition == 'test':
                return tf.data.Dataset.from_tensor_slices((
                    list(map(word_to_indices_poets, data_dict[client_id]['x'])),
                    list(map(letter_to_index_poets, data_dict[client_id]['y'])))
                )
            else:
                if partition == 'train':
                    return tf.data.Dataset.from_tensor_slices((
                        list(map(word_to_indices_poets, data_dict[client_id]['x'])),
                        list(map(letter_to_index_poets, data_dict[client_id]['y'])))
                    ).take(math.floor(len(data_dict[client_id]['x']) * (1 - self.cfg.val_fraction)))
                else:
                    return tf.data.Dataset.from_tensor_slices((
                        list(map(word_to_indices_poets, data_dict[client_id]['x'])),
                        list(map(letter_to_index_poets, data_dict[client_id]['y'])))
                    ).skip(math.floor(len(data_dict[client_id]['x']) * (1 - self.cfg.val_fraction)))

        # Train/Val data
        client_ids, cluster_ids, train_data = self._read_data_from_dir(os.path.join(self.cfg.data_dir, 'train'))
        avg_n_classes_per_client = np.mean([len(np.unique(train_data[client_id]['y'])) for client_id in client_ids])
        logging.info(f'Avg. classes per client: {avg_n_classes_per_client}')
        self.train_ds = ClientData.from_clients_and_fn(client_ids,
                                                       partial(create_tf_dataset_for_client_fn, train_data, 'train'))
        self.val_ds = ClientData.from_clients_and_fn(client_ids,
                                                     partial(create_tf_dataset_for_client_fn, train_data, 'val'))
        self.cluster_ids = {client_id: cluster_id for client_id, cluster_id in zip(client_ids, cluster_ids)}

        if self.cfg.num_clients != len(client_ids):
            logging.info(
                f'Number of clients in loaded data does not match the specified number in the config ({len(client_ids)} vs {self.cfg.num_clients}).')
            self.cfg.num_clients = len(client_ids)

        # Test data
        client_ids, _, test_data = self._read_data_from_dir(os.path.join(self.cfg.data_dir, 'test'))
        self.test_ds = ClientData.from_clients_and_fn(client_ids,
                                                      partial(create_tf_dataset_for_client_fn, test_data, 'test'))

        self._local_dataset_sizes = {
            client_id: (math.floor(len(train_data[client_id]['y']) * (1 - self.cfg.val_fraction)),
                        math.ceil(len(train_data[client_id]['y']) * self.cfg.val_fraction),
                        len(test_data[client_id]['y']))
            for client_id in client_ids}

        logging.info(self.train_ds.element_type_structure)