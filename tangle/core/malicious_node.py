import numpy as np
import tensorflow as tf

from tangle.core.node import Node, NodeConfiguration
from tangle.core.poison_type import PoisonType

FLIP_FROM_CLASS = 3
FLIP_TO_CLASS = 8


class MaliciousNode(Node):
    def __init__(self, tangle, tx_store, tip_selector, client_id, cluster_id, data, model=None, poison_type=PoisonType.Disabled, config=NodeConfiguration()):
        self.poison_type = poison_type

        if self.poison_type == PoisonType.LabelFlip:
            def flip_labels(x_batch, y_batch):
                flip_forward = tf.equal(y_batch, FLIP_FROM_CLASS)
                flip_backward = tf.equal(y_batch, FLIP_TO_CLASS)
                y_batch = tf.where(flip_forward, tf.ones_like(y_batch) * FLIP_TO_CLASS, y_batch)
                y_batch = tf.where(flip_backward, tf.ones_like(y_batch) * FLIP_FROM_CLASS, y_batch)

                return x_batch, y_batch

            data['train'] = data['train'].map(flip_labels)
            data['val'] = data['val'].map(flip_labels)
            #data['test'] = data['test'].map(flip_labels)

        super().__init__(tangle, tx_store, tip_selector, client_id, cluster_id, data, model=model, config=config)

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
