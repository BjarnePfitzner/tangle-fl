import random
import ray

from ..core.tip_selection.accuracy_tip_selector import AccuracyTipSelectorSettings
from ..lab import TipSelectorFactory, Lab
from ..models.baseline_constants import ACCURACY_KEY

from .ray_accuracy_tip_selector import RayAccuracyTipSelector

class RayTipSelectorFactory(TipSelectorFactory):
    def create(self, tangle, dataset, client_id, tx_store):
        if self.config.tip_selector == 'accuracy':
            tip_selection_settings = {}
            tip_selection_settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] = self.config.acc_tip_selection_strategy
            tip_selection_settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS] = self.config.acc_cumulate_ratings
            tip_selection_settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] = self.config.acc_ratings_to_weights
            tip_selection_settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] = self.config.acc_select_from_weights
            tip_selection_settings[AccuracyTipSelectorSettings.ALPHA] = self.config.acc_alpha

            futures = [self.compute_accuracy_ratings.remote(self, client_id, tx_id, random.randint(0, 4294967295), dataset.model_config, dataset.remote_train_data[client_id], tx_store) for tx_id, _ in tangle.transactions.items()]

            return RayAccuracyTipSelector(tangle, tip_selection_settings, {tx_id: r for tx_id, r in ray.get(futures)})

        return super().create(tangle)

    @ray.remote
    def compute_accuracy_ratings(self, node_id, tx_id, seed, model_config, data, tx_store):
        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        node_model = Lab.create_client_model(seed, model_config)
        node_model.set_params(tx_store.load_transaction_weights(tx_id))

        data = { 'x': ray.get(data['x']), 'y': ray.get(data['y']) }

        return tx_id, node_model.test(data)[ACCURACY_KEY]
