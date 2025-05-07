from functools import reduce, partial
from abc import ABC, abstractmethod
import logging
from copy import copy
import math
import os
import json

import wandb
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow_federated.python.simulation.datasets import ClientData

from data.dataset_type import DatasetType


class AbstractDataset(ABC):
    def __init__(self, dataset_cfg, normalisation_mean_zero=False, centralised=False):
        self.cfg = dataset_cfg
        self.normalisation_mean_zero = normalisation_mean_zero
        self.centralised = centralised

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.is_preprocessed = False
        self.cluster_ids = {}
        self._local_dataset_sizes = None

        logging.info(f'Loading {"centralised " if centralised else ""}{self.name} dataset...')
        if centralised:
            self._load_tf_dataset()
            self.client_ids = ['central_client']
        else:
            if self.cfg.get('data_dir') is not None:
                self._load_tff_dataset_from_files()
            else:
                self._load_tff_dataset()
            self.client_ids = copy(self.train_ds.client_ids)
        self.print_dataset_sizes()

    @property
    def name(self):
        return self.cfg.name

    @property
    def type(self):
        return DatasetType.from_value(self.cfg.name)

    @property
    @abstractmethod
    def dataset_size(self):
        pass

    @property
    def avg_local_dataset_size(self):
        return math.floor(self.dataset_size['train'] / self.cfg.num_clients), 0.0

    @property
    @abstractmethod
    def class_labels(self):
        pass

    @abstractmethod
    def get_default_accuracy(self):
        pass

    @staticmethod
    def _read_data_from_dir(data_dir):
        clients = []
        clusters = []
        data = {}

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as json_file:
                cdata = json.load(json_file)
            clients.extend(cdata['users'])
            if 'cluster_ids' in cdata:
                clusters.extend(cdata['cluster_ids'])
            data.update(cdata['user_data'])

        return clients, clusters, data

    def get_dataset_size_for_client(self, client_id):
        assert not self.centralised
        return self.avg_local_dataset_size[0]

    def get_all_dataset_partitions_for_client(self, client_id):
        return {
           'train': self.train_ds.create_tf_dataset_for_client(client_id),
           'val': self.val_ds.create_tf_dataset_for_client(client_id),
           'test': self.test_ds.create_tf_dataset_for_client(client_id)
        }

    def get_cluster_id_for_client(self, client_id):
        return self.cluster_ids.get(client_id, -1)

    def _load_tf_dataset(self):
        self._load_tff_dataset()
        self.train_ds = self.train_ds.create_tf_dataset_from_all_clients()
        self.val_ds = self.val_ds.create_tf_dataset_from_all_clients()
        self.test_ds = self.test_ds.create_tf_dataset_from_all_clients()

    def _load_tff_dataset(self):
        self._load_tf_dataset()

        if self.cfg.class_distribution == 'iid':
            self.train_ds = self._create_federated_dataset(self.train_ds, self.cfg.num_clients)
            self.val_ds = self._create_federated_dataset(self.val_ds, self.cfg.num_clients)
            self.test_ds = self._create_federated_dataset(self.test_ds, self.cfg.num_clients)

        elif self.cfg.class_distribution.endswith('-class-nonIID'):
            self.train_ds, self.val_ds = self._prepare_non_iid_data(self.train_ds)  # todo still broken
        else:
            raise NotImplementedError('Only iid distribution implemented so far')

    def _load_tff_dataset_from_files(self):
        # Load data from files
        logging.info('Loading data from files...')

        def create_tf_dataset_for_client_fn(data_dict, partition, client_id):
            def _reshape(x):
                if self.cfg.get('data_ch') is None:
                    return tf.reshape(x, (self.cfg.data_dim, ))
                return tf.reshape(x, (self.cfg.data_dim, self.cfg.data_dim, self.cfg.data_ch))
            if partition == 'test':
                return tf.data.Dataset.from_tensor_slices(
                    (list(map(_reshape, data_dict[client_id]['x'])), data_dict[client_id]['y'])
                )
            else:
                if partition == 'train':
                    return tf.data.Dataset.from_tensor_slices(
                        (list(map(_reshape, data_dict[client_id]['x'])), data_dict[client_id]['y'])
                    ).take(math.floor(len(data_dict[client_id]['x']) * (1 - self.cfg.val_fraction)))
                else:
                    return tf.data.Dataset.from_tensor_slices(
                        (list(map(_reshape, data_dict[client_id]['x'])), data_dict[client_id]['y'])
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

        self._local_dataset_sizes = {client_id: (math.floor(len(train_data[client_id]['y']) * (1 - self.cfg.val_fraction)),
                                                math.ceil(len(train_data[client_id]['y']) * self.cfg.val_fraction),
                                                len(test_data[client_id]['y']))
                                    for client_id in client_ids}

    def preprocess_datasets(self, train_batch_size, test_batch_size, local_epochs):
        if self.centralised:
            self.train_ds = self._preprocess_tf_dataset(self.train_ds, train_batch_size)
            self.val_ds = self._preprocess_tf_dataset(self.val_ds, test_batch_size)
            self.test_ds = self._preprocess_tf_dataset(self.test_ds, test_batch_size)
        else:
            self.train_ds = self._preprocess_tff_dataset(self.train_ds, train_batch_size, local_epochs)
            self.val_ds = self._preprocess_tff_dataset(self.val_ds, test_batch_size, 1)
            self.test_ds = self._preprocess_tff_dataset(self.test_ds, test_batch_size, 1)

        self.is_preprocessed = True

    @classmethod
    def _preprocess_tff_dataset(cls, dataset, batch_size, local_epochs, drop_remainder=False):
        def preprocess_fn(ds):
            return (ds
                    .repeat(local_epochs)
                    .batch(batch_size, drop_remainder=drop_remainder)
                    .prefetch(AUTOTUNE))

        return dataset.preprocess(preprocess_fn)

    @classmethod
    def _preprocess_tf_dataset(cls, dataset, batch_size, drop_remainder=False):
        return (dataset
                .batch(batch_size, drop_remainder=drop_remainder)
                .prefetch(AUTOTUNE))

    @classmethod
    def _create_federated_dataset(cls, dataset, n_clients):
        def create_tf_dataset_for_client_fn(client_id):
            the_id = int(client_id[client_id.find('_') + 1:])
            return dataset.shard(num_shards=n_clients, index=the_id)

        return ClientData.from_clients_and_fn([f'client_{str(the_id)}' for the_id in range(n_clients)],
                                              create_tf_dataset_for_client_fn)

    def _create_single_class_datasets(self, dataset):
        single_class_datasets = {}
        for label in range(self.cfg.n_classes):
            single_class_datasets[label] = dataset.filter(lambda _, y: y == label)
        return single_class_datasets

    def _prepare_non_iid_data(self, dataset):
        n_classes_per_client = int(self.cfg.class_distribution[0])
        single_class_datasets = self._create_single_class_datasets(dataset)
        n_clients_per_class = int(
            self.cfg.num_clients * n_classes_per_client / self.cfg.n_classes)
        while True:
            try:
                client_budget_for_class = {label: n_clients_per_class for label in range(self.cfg.n_classes)}
                classes_for_client = {}
                for i in range(self.cfg.num_clients):
                    available_classes = list(client_budget_for_class.keys())
                    selected_classes = np.random.choice(available_classes, n_classes_per_client,
                                                        replace=False,
                                                        p=[client_budget_for_class[c] / sum(
                                                            client_budget_for_class.values())
                                                           for c in client_budget_for_class.keys()])
                    classes_for_client[f'client_{i}'] = []  # holds information in the form (class, shard_id)
                    for selected_class in selected_classes:
                        client_budget_for_class[selected_class] -= 1
                        classes_for_client[f'client_{i}'].append(
                            (selected_class, client_budget_for_class[selected_class]))
                        if client_budget_for_class[selected_class] == 0:
                            del client_budget_for_class[selected_class]
            except ValueError:
                continue
            else:
                break

        def prepare_single_client_data(client_id):
            datasets = []
            for label, shard_id in classes_for_client[client_id]:
                datasets.append(single_class_datasets[label].shard(num_shards=n_clients_per_class,
                                                                   index=shard_id))
            return reduce(lambda x, y: x.concatenate(y), datasets).shuffle(
                buffer_size=int(self.dataset_size['train'] / self.cfg.num_clients)).cache()

        return ClientData.from_clients_and_fn([f'client_{str(the_id)}' for the_id in range(self.cfg.num_clients)],
                                              prepare_single_client_data)

    def log_sample_data(self):
        if self.is_preprocessed:
            logging.info('Not logging sample data since data has been preprocessed already.')
            return

        logging.info('Logging sample data')
        sample_data = np.array(list(map(lambda x: x[0], self.test_ds.take(16).as_numpy_iterator())))
        plot_grid_image(images=sample_data,
                        wandb_name='Sample Images',
                        image_size=self.cfg.data_dim,
                        wandb_commit=True)

    def print_dataset_sizes(self):
        logging.info(f'Dataset sizes: {self.dataset_size}')
        if not self.centralised:
            logging.info(f'Number of clients: {len(self.client_ids)}')
            logging.info(f'Local train sizes: {round(self.avg_local_dataset_size[0], 2)} +/- {round(self.avg_local_dataset_size[1], 2)}')

    def select_clients(self, my_round, client_fraction=0.01, sample_clients=False, log_number_of_clients=True, active_clients=None):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Returns:
            list of (client_id, cluster_id)
        """
        np.random.seed(my_round)
        if active_clients is None:
            active_clients = self.client_ids
        if sample_clients:
            x = np.random.uniform(size=len(active_clients))
            clients = [active_clients[i] for i in range(len(active_clients)) if x[i] < client_fraction]
            if len(clients) == 0:
                clients = list(np.random.choice(active_clients, 1))
            if log_number_of_clients:
                wandb.log({'train/num_clients': len(clients)}, step=my_round)
            return clients
        else:
            num_clients = int(len(active_clients) * client_fraction)
            client_indices = np.random.choice(range(len(active_clients)), num_clients, replace=False)
            if log_number_of_clients:
                wandb.log({'train/num_clients': num_clients}, step=my_round)
            return [active_clients[i] for i in client_indices]


def plot_grid_image(images, image_size, wandb_name, wandb_commit):
    import matplotlib.pyplot as plt

    grid_dim = np.math.ceil(np.sqrt(images.shape[0]))
    fig = plt.figure(figsize=(image_size / 100 * grid_dim, image_size / 100 * grid_dim), dpi=300)

    for i in range(images.shape[0]):
        plt.subplot(grid_dim, grid_dim, i + 1)
        if images.shape[3] == 1:
            # plt.imshow(images[i, :, :, 0], vmin=vmin, vmax=1.0, cmap='gray')
            plt.imshow(images[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(images[i])
        plt.axis('off')

    wandb.log({wandb_name: fig}, commit=wandb_commit)
    plt.close(fig)
