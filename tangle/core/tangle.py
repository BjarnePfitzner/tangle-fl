import random

import numpy as np

from .transaction import Transaction

class Tangle:
    def __init__(self, transactions, genesis):
        self.transactions = transactions
        self.genesis = genesis

    def add_transaction(self, tip):
        self.transactions[tip.name()] = tip

    def test_model(self, test_fn, clients_to_test, tip_selection_settings, set_to_use='test'):
        metrics = {}

        test_params = [[client.id, client.cluster_id, client.group, client.model.flops, random.randint(0, 4294967295), client.train_data, client.eval_data, self.name, set_to_use, tip_selection_settings] for client in clients_to_test]
        results = self.process_pool.starmap(test_fn, test_params)

        for client, c_metrics in results:
            metrics[client] = c_metrics

        return metrics
