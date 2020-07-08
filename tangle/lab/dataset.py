import os
import itertools
import numpy as np

from ..models.utils.model_utils import read_data

class Dataset:
    def __init__(self, lab_config, model_config):
        self.lab_config = lab_config
        self.model_config = model_config
        self.clients, self.train_data, self.test_data = self.setup_clients()

    def setup_clients(self):
        eval_set = 'test' if not self.lab_config.use_val_set else 'val'
        train_data_dir = os.path.join(self.lab_config.model_data_dir, 'train')
        test_data_dir = os.path.join(self.lab_config.model_data_dir, eval_set)

        users, cluster_ids, train_data, test_data = read_data(train_data_dir, test_data_dir)

        return list(itertools.zip_longest(users, cluster_ids)), train_data, test_data

    def select_clients(self, my_round, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (client_id, cluster_id)
        """
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(my_round)
        client_indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return [self.clients[i] for i in client_indices]
