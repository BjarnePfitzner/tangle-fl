import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import ray

from ..lab import Lab, parse_args
from ..lab.config import LabConfiguration, ModelConfiguration, RunConfiguration

from .ray_transaction_store import RayTransactionStore

class RayLab(Lab):
    def __init__(self, config, model_config):
        super().__init__(config, model_config, tx_store=RayTransactionStore(config.tangle_dir))

        self.remote_train_data = {
            cid : ray.put(data) for (cid, data) in self.train_data.items()
        }
        self.remote_test_data = {
            cid : ray.put(data) for (cid, data) in self.test_data.items()
        }

    def create_genesis(self):

        @ray.remote
        def _create_genesis(self):
            return super().create_genesis()

        return ray.get(_create_genesis.remote(self))

    def create_node_transactions(self, tangle, round, clients):
        @ray.remote
        def _create_node_transaction(tangle, round, client_id, cluster_id, train_data, eval_data, seed, model_config, tx_store):
            import tensorflow as tf

            # Suppress tf warnings
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

            return Lab.create_node_transaction(tangle, round, client_id, cluster_id, train_data, eval_data, seed, model_config, tx_store)

        futures = [_create_node_transaction.remote(tangle, round, client_id, cluster_id, self.remote_train_data[client_id], self.remote_test_data[client_id], random.randint(0, 4294967295), self.model_config, self.tx_store)
                   for (client_id, cluster_id) in clients]

        return ray.get(futures)

    def validate_nodes(self, tangle, clients):
        @ray.remote
        def _test_single(tangle, round, client_id, cluster_id, train_data, eval_data, seed, model_config, tx_store, set_to_use):
            return Lab.test_single(tangle, round, client_id, cluster_id, train_data, eval_data, seed, model_config, tx_store, set_to_use)

        futures = [_test_single.remote(tangle, round, client_id, cluster_id, self.remote_train_data[client_id], self.remote_test_data[client_id], random.randint(0, 4294967295), self.model_config, self.tx_store, 'test')
                   for (client_id, cluster_id) in clients]

        return ray.get(futures)

def main():
    run_config, lab_config, model_config = parse_args(RunConfiguration, LabConfiguration, ModelConfiguration)

    ray.init(webui_host='0.0.0.0')

    lab = RayLab(lab_config, model_config)

    lab.train(run_config.clients_per_round, run_config.start_from_round, run_config.num_rounds)
    lab.validate(run_config.num_rounds-1)
