import random
import ray

from tangle.core.tip_selection import LazyAccuracyTipSelector
from tangle.lab.lab import Lab
from tangle.models.baseline_constants import ACCURACY_KEY

class RayLazyAccuracyTipSelector(LazyAccuracyTipSelector):
    def __init__(self, tangle, tip_selection_settings, particle_settings, dataset):
        super().__init__(tangle, tip_selection_settings, particle_settings)
        self.dataset = dataset

    def _get_or_calculate_accuracies(self, txs_to_eval, node, future_set_cache):
        futures = [self._compute_accuracy_rating.remote(self, node.id, tx_id, random.randint(0, 4294967295), self.dataset.model_config, self.dataset.remote_train_data[node.id], node.tx_store, future_set_cache) for tx_id in txs_to_eval]
        return ray.get(futures)

    @ray.remote
    def _compute_accuracy_rating(self, node_id, tx_id, seed, model_config, data, tx_store, future_set_cache):

        if node_id in self.accuracies:
            if tx_id in self.accuracies[node_id]:
                return tx_id, self.accuracies[node_id][tx_id]

        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        node_model = Lab.create_client_model(seed, model_config)
        node_model.set_params(tx_store.load_transaction_weights(tx_id))

        data = { 'x': ray.get(data['x']), 'y': ray.get(data['y']) }
        accuracy = node_model.test(data)[ACCURACY_KEY]
        # Multiply size of tree behind this node (see `_compute_ratings` in accuracy_tip_selector.py)
        accuracy *= len(super().future_set(tx_id, self.approving_transactions, future_set_cache)) + 1

        return accuracy
