import random
import ray

from ..lab import Lab

from ..core.tip_selection import AccuracyTipSelector
from ..models.baseline_constants import ACCURACY_KEY


@ray.remote
def _compute(node_id, tx_id, seed, model_config, data, tx_store):
    import tensorflow as tf

    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    node_model = Lab.create_client_model(seed, model_config)
    node_model.set_params(tx_store.load_transaction_weights(tx_id))
    return tx_id, node_model.test(data)[ACCURACY_KEY]

class RayAccuracyTipSelector(AccuracyTipSelector):
    def __init__(self, tangle, settings, dataset):
        super().__init__(tangle, settings)
        self.dataset = dataset

    def _compute_ratings(self, node):
        futures = [_compute.remote(node.id, tx_id, random.randint(0, 4294967295), self.dataset.model_config, self.dataset.remote_train_data[node.id], node.tx_store) for tx_id, _ in self.tangle.transactions.items()]
        return {tx_id: r for tx_id, r in ray.get(futures)}
