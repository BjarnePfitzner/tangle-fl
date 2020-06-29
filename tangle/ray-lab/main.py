import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray

from ..lab import Lab, LabConfiguration, ModelConfiguration, parse_args

class RayLab(Lab):
    def __init__(self, config, model_config):
        super().__init__(config, model_config)

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

        futures = [_create_node_transaction.remote(tangle, round, client_id, cluster_id, self.remote_train_data[client_id], self.remote_test_data[client_id], self.config.seed, self.model_config, self.tx_store)
                   for (client_id, cluster_id) in clients]

        return ray.get(futures)

def main():
    args = parse_args()

    ray.init(webui_host='0.0.0.0')

    config = LabConfiguration(
        args.seed,
        args.model_data_dir,
        args.tangle_dir,
        args.tangle_tx_dir
    )

    model_config = ModelConfiguration(
        args.dataset,
        args.model,
        args.lr,
        args.use_val_set,
        args.num_epochs,
        args.batch_size
    )

    lab = RayLab(config, model_config)

    lab.train(args.clients_per_round, args.start_from_round, args.num_rounds)
