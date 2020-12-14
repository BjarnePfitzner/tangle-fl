import numpy as np
import sys

from .node import Node, NodeConfiguration
from .poison_type import PoisonType

class MaliciousNode(Node):
    def __init__(self, tangle, tx_store, tip_selector, client_id, cluster_id, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None, poison_type=PoisonType.Disabled, config=NodeConfiguration()):
        self.poison_type = poison_type
        super().__init__(tangle, tx_store, tip_selector, client_id, cluster_id, train_data, eval_data, model=None, config=config)

    def train(self, model_params):
        if self.poison_type == PoisonType.Random:
            malicious_weights = [np.random.RandomState().normal(size=w.shape) for w in model_params]
            return malicious_weights
        else:
            return super().train(model_params)


    def create_transaction(self):
        t, weights = super().create_transaction()

        if t is not None and self.poison_type != PoisonType.Disabled:
            t.add_metadata('poisoned', True)

        return t, weights
