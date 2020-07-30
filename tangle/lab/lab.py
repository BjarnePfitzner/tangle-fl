import random
import numpy as np
import importlib
import itertools

from ..models.baseline_constants import MODEL_PARAMS, ACCURACY_KEY
from ..core import Tangle, Transaction, Node
from .lab_transaction_store import LabTransactionStore


class Lab:
    def __init__(self, tip_selector_factory, config, model_config, tx_store=None):
        self.tip_selector_factory = tip_selector_factory
        self.config = config
        self.model_config = model_config
        self.tx_store = tx_store if tx_store is not None else LabTransactionStore(self.config.tangle_dir)

        # Set the random seed if provided (affects client sampling, and batching)
        random.seed(1 + config.seed)
        np.random.seed(12 + config.seed)

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

        model = ClientModel(seed, *model_params)
        model.num_epochs = model_config.num_epochs
        model.batch_size = model_config.batch_size
        return model

    def create_genesis(self):
        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        client_model = self.create_client_model(self.config.seed, self.model_config)

        genesis = Transaction([])
        genesis.add_metadata('time', 0)
        self.tx_store.save(genesis, client_model.get_params())

        return genesis

    def create_node_transaction(self, tangle, round, client_id, cluster_id, train_data, eval_data, seed, model_config, tip_selector, tx_store):

        client_model = Lab.create_client_model(seed, model_config)
        node = Node(tangle, tx_store, tip_selector, client_id, cluster_id, train_data, eval_data, client_model)
        tx, tx_weights = node.create_transaction()

        if tx is not None:
            tx.add_metadata('time', round)

        return tx, tx_weights

    def create_node_transactions(self, tangle, round, clients, dataset):
        tip_selector = self.tip_selector_factory.create(tangle)

        result = [self.create_node_transaction(tangle, round, client_id, cluster_id, dataset.train_data[client_id], dataset.test_data[client_id], self.config.seed, self.model_config, tip_selector, self.tx_store)
                  for (client_id, cluster_id) in clients]

        for tx, tx_weights in result:
            if tx is not None:
                self.tx_store.save(tx, tx_weights)

        return [tx for tx, _ in result]

    def create_malicious_transaction(self):
        pass

    def train(self, num_nodes, start_from_round, num_rounds, eval_every, dataset):
        if num_rounds == -1:
            rounds_iter = itertools.count(start_from_round)
        else:
            rounds_iter = range(start_from_round, num_rounds)

        if start_from_round > 0:
            tangle = self.tx_store.load_tangle(0)

        for round in rounds_iter:
            print('Started training for round %s' % round)

            if round == 0:
                genesis = self.create_genesis()
                tangle = Tangle({genesis.id: genesis}, genesis.id)
            else:
                clients = dataset.select_clients(round, num_nodes)

                for tx in self.create_node_transactions(tangle, round, clients, dataset):
                    if tx is not None:
                        tangle.add_transaction(tx)

            self.tx_store.save_tangle(tangle, round)

            if round % eval_every == 0:
                self.print_validation_results(self.validate(round, dataset), mode='avg')

    def test_single(self, tangle, client_id, cluster_id, train_data, eval_data, seed, set_to_use, tip_selector):
        import tensorflow as tf

        random.seed(1 + seed)
        np.random.seed(12 + seed)
        tf.compat.v1.set_random_seed(123 + seed)

        client_model = self.create_client_model(seed, self.model_config)
        node = Node(tangle, self.tx_store, tip_selector, client_id, cluster_id, train_data, eval_data, client_model)

        reference_txs, reference = node.obtain_reference_params()
        metrics = node.test(reference, set_to_use)

        return metrics

    def validate_nodes(self, tangle, clients, dataset):
        tip_selector = self.tip_selector_factory.create(tangle)
        return [self.test_single(tangle, client_id, cluster_id, dataset.train_data[client_id], dataset.test_data[client_id], random.randint(0, 4294967295), 'test', tip_selector) for client_id, cluster_id in clients]

    def validate(self, round, dataset):
        print('Validate for round %s' % round)
        tangle = self.tx_store.load_tangle(round)
        client_indices = np.random.choice(range(len(dataset.clients)), min(int(len(dataset.clients) * 0.1), len(dataset.clients)), replace=False)
        validation_clients = [dataset.clients[i] for i in client_indices]
        return self.validate_nodes(tangle, validation_clients, dataset)

    def print_validation_results(self, results, mode='avg'):
        avg_acc = np.average([r[ACCURACY_KEY] for r in results])
        avg_loss = np.average([r['loss'] for r in results])

        avg_message = 'Average %s: %s\nAverage loss: %s' % (ACCURACY_KEY, avg_acc, avg_loss)

        if mode == 'avg':
            print(avg_message)
        if mode == 'all':
            print(avg_message)
            print(results)
