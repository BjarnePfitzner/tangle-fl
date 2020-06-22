import json
import os
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

    def save(self, tangle_data_dir, tangle_name):
        os.makedirs(tangle_data_dir, exist_ok=True)

        n = [{'name': t.name(),
              'time': t.tag,
              'malicious': t.malicious,
              'parents': list(t.parents),
              'issuer': t.client_id,
              'clusterId': t.cluster_id } for _, t in self.transactions.items()]

        with open(f'{tangle_data_dir}/tangle_{tangle_name}.json', 'w') as outfile:
            json.dump({'nodes': n, 'genesis': self.genesis}, outfile)

        self.name = tangle_name

    @classmethod
    def fromfile(cls, tangle_data_dir, tangle_name):
        with open(f'{tangle_data_dir}/tangle_{tangle_name}.json', 'r') as tanglefile:
            t = json.load(tanglefile)

        transactions = {n['name']: Transaction(
                                        None,
                                        set(n['parents']),
                                        n['issuer'],
                                        n['clusterId'],
                                        n['name'],
                                        n['time'],
                                        n['malicious'] if 'malicious' in n else False
                                    ) for n in t['nodes']}
        tangle = cls(transactions, t['genesis'])
        tangle.name = tangle_name
        return tangle
