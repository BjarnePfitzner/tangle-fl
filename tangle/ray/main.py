import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import ray

from ..lab import Dataset, Lab, TipSelectorFactory, parse_args
from ..lab.config import LabConfiguration, ModelConfiguration, PoisoningConfiguration, RunConfiguration, TangleConfiguration, TipSelectorConfiguration
from ..core.tip_selection.accuracy_tip_selector import AccuracyTipSelectorSettings

from .ray_transaction_store import RayTransactionStore
from .ray_accuracy_tip_selector import RayAccuracyTipSelector

class RayTipSelectorFactory(TipSelectorFactory):
    def create(self):
        if self.config.tip_selector == 'accuracy':
            tip_selection_settings = {}
            tip_selection_settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] = self.config.acc_tip_selection_strategy
            tip_selection_settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS] = self.config.acc_cumulate_ratings
            tip_selection_settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] = self.config.acc_ratings_to_weights
            tip_selection_settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] = self.config.acc_select_from_weights
            tip_selection_settings[AccuracyTipSelectorSettings.ALPHA] = self.config.acc_alpha

            return RayAccuracyTipSelector(self.tangle, tip_selection_settings, self.dataset)

        return super().create()

    @classmethod
    def factory(cls, dataset):
        class _RayTipSelectorFactory(RayTipSelectorFactory):
            def __init__(self, config, tangle):
                super().__init__(config, tangle)
                self.dataset = dataset

        return _RayTipSelectorFactory

class RayDataset(Dataset):
    def __init__(self, lab_config, model_config):
        super().__init__(lab_config, model_config)

        self.remote_train_data = {
            cid : ray.put(data) for (cid, data) in self.train_data.items()
        }
        self.remote_test_data = {
            cid : ray.put(data) for (cid, data) in self.test_data.items()
        }

class RayLab(Lab):
    def __init__(self, TipSelectorFactory, config, model_config, tip_selector_config):
        super().__init__(TipSelectorFactory, config, model_config, tip_selector_config, tx_store=RayTransactionStore(config.tangle_dir))

    def create_genesis(self):
        @ray.remote
        def _create_genesis(self):
            return super().create_genesis()

        genesis = ray.get(_create_genesis.remote(self))

        # Cache it
        self.tx_store.load_transaction_weights(genesis.id)

        return genesis

    @ray.remote
    def create_node_transaction(self, tangle, round, client_id, cluster_id, train_data, eval_data, seed):
        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        return super().create_node_transaction(tangle, round, client_id, cluster_id, train_data, eval_data, seed, self.model_config, self.tip_selector_config, self.tx_store)

    def create_node_transactions(self, tangle, round, clients, dataset):

        futures = [self.create_node_transaction.remote(self, tangle, round, client_id, cluster_id, dataset.remote_train_data[client_id], dataset.remote_test_data[client_id], random.randint(0, 4294967295))
                   for (client_id, cluster_id) in clients]

        result = ray.get(futures)

        for tx, tx_weights in result:
            if tx is not None:
                self.tx_store.save(tx, tx_weights)

        return [tx for tx, _ in result]

    @ray.remote
    def test_single(self, tangle, client_id, cluster_id, train_data, eval_data, seed, set_to_use):
        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        super().test_single(tangle, client_id, cluster_id, train_data, eval_data, seed, set_to_use)

    def validate_nodes(self, tangle, clients, dataset):

        futures = [self.test_single.remote(self, tangle, client_id, cluster_id, dataset.remote_train_data[client_id], dataset.remote_test_data[client_id], random.randint(0, 4294967295), 'test')
                   for (client_id, cluster_id) in clients]

        return ray.get(futures)

def main():
    run_config, lab_config, model_config, poisoining_config, tangle_config, tip_selector_config = \
        parse_args(RunConfiguration, LabConfiguration, ModelConfiguration, PoisoningConfiguration, TangleConfiguration, TipSelectorConfiguration)

    ray.init(webui_host='0.0.0.0')

    dataset = RayDataset(lab_config, model_config)

    # Ürgh
    MyLab = RayLab.factory(RayTipSelectorFactory.factory(dataset))
    lab = MyLab(lab_config, model_config, tip_selector_config)

    lab.train(run_config.clients_per_round, run_config.start_from_round, run_config.num_rounds, dataset)
    lab.validate(run_config.num_rounds-1, dataset)
