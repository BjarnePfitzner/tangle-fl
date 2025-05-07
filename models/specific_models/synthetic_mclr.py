"""Bag-of-words logistic regression."""

import tensorflow as tf
from tensorflow.keras import layers

from models.model import Model

class ClientModel(Model):

    def __init__(self, seed, lr, cfg, prox_mu=0):
        self.num_classes = cfg.num_classes
        self.data_dim = cfg.data_dim
        super(ClientModel, self).__init__(seed, lr, prox_mu=prox_mu)

    def create_model(self):
        model = tf.keras.models.Sequential(
            [
                layers.Dense(units=self.num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                             activation=tf.nn.softmax, input_shape=(self.data_dim,))
            ]
        )

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        return model, loss
