import os
import random
import numpy as np
import importlib
import itertools

from ..models.baseline_constants import MODEL_PARAMS
from ..models.metrics.writer import print_metrics
from ..core import Tangle, Transaction, Node
from ..core.tip_selection import TipSelector
from ..models.utils.model_utils import read_data
from .lab_transaction_store import LabTransactionStore


class Lab:
    def __init__(self, config, model_config):
        self.config = config
        self.model_config = model_config

        # Set the random seed if provided (affects client sampling, and batching)
        random.seed(1 + config.seed)
        np.random.seed(12 + config.seed)

        self.clients, self.train_data, self.test_data = self.setup_clients(self.model_config.dataset)

        self.tx_store = LabTransactionStore(self.config.tangle_dir, self.config.tangle_tx_dir)

    def setup_clients(self, dataset):
        eval_set = 'test' if not self.model_config.use_val_set else 'val'
        train_data_dir = os.path.join(self.config.model_data_dir, dataset, 'data', 'train')
        test_data_dir = os.path.join(self.config.model_data_dir, dataset, 'data', eval_set)

        users, cluster_ids, train_data, test_data = read_data(train_data_dir, test_data_dir)

        return list(itertools.zip_longest(users, cluster_ids)), train_data, test_data

    @staticmethod
    def create_client_model(seed, model_config):
        model_path = '.%s.%s' % (model_config.dataset, model_config.model)
        mod = importlib.import_module(model_path, package='tangle.models')
        ClientModel = getattr(mod, 'ClientModel')

        # Create 2 models
        model_params = MODEL_PARAMS['%s.%s' % (model_config.dataset, model_config.model)]
        if model_config.lr != -1:
            model_params_list = list(model_params)
            model_params_list[0] = model_config.lr
            model_params = tuple(model_params_list)

        return ClientModel(seed, *model_params)

    def create_genesis(self):
        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        client_model = self.create_client_model(self.config.seed, self.model_config)

        genesis = Transaction([], "", None, tag=0)
        self.tx_store.save(genesis, client_model.get_params())

        return genesis

    @staticmethod
    def create_node_transaction(tangle, round, client_id, cluster_id, train_data, eval_data, seed, model_config, tx_store):

        # import tensorflow as tf

        # random.seed(1 + seed)
        # np.random.seed(12 + seed)
        # tf.compat.v1.set_random_seed(123 + seed)

        client_model = Lab.create_client_model(seed, model_config)
        tip_selector = TipSelector(tangle)
        node = Node(tangle, tx_store, tip_selector, client_id, cluster_id, train_data, eval_data, client_model)
        tx, tx_weights = node.create_transaction(model_config.num_epochs, model_config.batch_size)

        if tx is not None:
            tx_store.save(tx, tx_weights)
            return tx

        return None

    def create_node_transactions(self, tangle, round, clients):
        result = [self.create_node_transaction(tangle, round, client_id, cluster_id, self.train_data[client_id], self.test_data[client_id], self.config.seed, self.model_config, self.tx_store)
                  for (client_id, cluster_id) in clients]

        return result

    def create_malicious_transaction(self):
        pass

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (client_id, cluster_id)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        client_indices = np.random.choice(range(len(possible_clients)), num_clients, replace=False)
        return [self.clients[i] for i in client_indices]

    def train(self, num_nodes, start_from_round, num_rounds):
        if num_rounds == -1:
            rounds_iter = itertools.count(start_from_round)
        else:
            rounds_iter = range(start_from_round, num_rounds)

        if start_from_round > 0:
            tangle = self.tx_store.load_tangle(0)

        for round in rounds_iter:
            if round == 0:
                genesis = self.create_genesis()
                tangle = Tangle({genesis.name(): genesis}, genesis.name())
            else:
                clients = self.select_clients(round, self.clients, num_nodes)

                for tx in self.create_node_transactions(tangle, round, clients):
                    if tx is not None:
                        tangle.add_transaction(tx)

            self.tx_store.save_tangle(tangle, round)


    @staticmethod
    def test_single(tangle, round, client_id, cluster_id, train_data, eval_data, seed, model_config, tx_store, set_to_use):
        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        random.seed(1 + seed)
        np.random.seed(12 + seed)
        tf.compat.v1.set_random_seed(123 + seed)

        client_model = Lab.create_client_model(seed, model_config)
        tip_selector = TipSelector(tangle)
        node = Node(tangle, tx_store, tip_selector, client_id, cluster_id, train_data, eval_data, client_model)

        reference_txs, reference = node.obtain_reference_params()
        node.model.set_params(reference)
        metrics = node.test(set_to_use)

        return metrics

    def validate(self, round):
        tangle = self.tx_store.load_tangle(round)
        client_indices = np.random.choice(range(len(self.clients)), min(int(len(self.clients) * 0.1), len(self.clients)), replace=False)
        validation_clients = [self.clients[i] for i in client_indices]
        for client_id, cluster_id in validation_clients:
            self.test_single(tangle, round, client_id, cluster_id, self.train_data[client_id], self.test_data[client_id], random.randint(0, 4294967295), self.model_config, self.tx_store, 'test')

    # def get_stat_writer_function(ids, groups, num_samples):
    #     def writer_fn(num_round, metrics, partition):
    #         print_metrics(
    #             num_round, ids, metrics, groups, num_samples, partition, 'leaf/models/metrics', '{}'.format('stat'))

    #     return writer_fn


    # def validate_old(self):
    #     tangle = self.tx_store.load_tangle(round)
    #     client_indices = np.random.choice(range(len(self.clients)), min(int(len(self.clients) * 0.1), len(self.clients)), replace=False)
    #     validation_clients = [self.clients[i] for i in client_indices]
    #     client_num_samples = {c.id: self. for c in validation_clients}
    #     stat_writer_fn = self.get_stat_writer_function(client_ids, client_groups, client_num_samples)
    #     self.print_stats(round, tangle, validation_clients, client_num_samples, stat_writer_fn, args.use_val_set, (poison_type != PoisonType.NONE), tip_selection_settings)

    # def print_stats(
    #     num_round, tangle, clients, num_samples, writer, use_val_set, tip_selection_settings):

    #     train_stat_metrics = {
    #         client_id: self.test_single(tangle, num_round, client_id, cluster_id, self.train_data[client_id], self.test_data[client_id], random.randint(0, 4294967295))
    #         for client_id, cluster_id in clients
    #     }
    #     print_metrics(train_stat_metrics, num_samples, num_round, prefix='train_')
    #     writer(num_round, train_stat_metrics, 'train')

    #     eval_set = 'test' if not use_val_set else 'val'
    #     test_stat_metrics = tangle.test_model(test_single, clients, tip_selection_settings, set_to_use=eval_set)
    #     average_test_metrics = print_metrics(test_stat_metrics, num_samples, num_round, prefix='{}_'.format(eval_set))
    #     writer(num_round, test_stat_metrics, eval_set)

    #     with open('%s/3_results.txt' % args.tangle_dir, 'a+') as file:
    #         file.write('Round %d \n' % num_round)
    #         file.write(str(average_test_metrics) + '\n')

    #     return average_test_metrics

    # def print_metrics(metrics, weights, num_round, prefix='', print_conf_matrix=False):
    #     """Prints weighted averages of the given metrics.

    #     Args:
    #         metrics: dict with client ids as keys. Each entry is a dict
    #             with the metrics of that client.
    #         weights: dict with client ids as keys. Each entry is the weight
    #             for that client.
    #     """
    #     ordered_weights = [weights[c] for c in sorted(weights) if c in metrics]
    #     metric_names = metrics_writer.get_metrics_names(metrics)
    #     to_ret = None
    #     average_metrics = {}
    #     for metric in metric_names:
    #         if metric == 'conf_matrix':
    #             continue
    #         ordered_metric = [metrics[c][metric] for c in sorted(metrics)]

    #         print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
    #             % (prefix + metric,
    #                 np.average(ordered_metric, weights=ordered_weights),
    #                 np.percentile(ordered_metric, 10),
    #                 np.percentile(ordered_metric, 50),
    #                 np.percentile(ordered_metric, 90)))
    #         average_metrics[metric] = np.average(ordered_metric, weights=ordered_weights)

    #     return average_metrics

    # def test_model(self, test_fn, clients_to_test, tip_selection_settings, set_to_use='test'):
    #     metrics = {}

    #     test_params = [[client.id, client.cluster_id, client.group, client.model.flops, random.randint(0, 4294967295), client.train_data, client.eval_data, self.name, set_to_use, tip_selection_settings] for client in clients_to_test]
    #     results = self.process_pool.starmap(test_fn, test_params)

    #     for client, c_metrics in results:
    #         metrics[client] = c_metrics

    #     return metrics
