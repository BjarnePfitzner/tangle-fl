"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import logging
from copy import deepcopy


def flatten(values):
    if isinstance(values[0], np.ndarray):
        return np.concatenate([value.flatten() for value in values],
                              axis=0)
    else:
        return tf.concat([tf.reshape(value, [-1]) for value in values],
                         axis=0)

class Model(ABC):
    num_classes: int

    def __init__(self, seed, lr, optimizer=None, prox_mu=0):
        self.lr = lr
        self.seed = seed
        self.prox_mu = prox_mu
        if prox_mu > 0:
            logging.debug(f'Using proximal term with mu={prox_mu}')
        self._optimizer = optimizer

        tf.random.set_seed(123 + seed)
        np.random.seed(seed)

        self.model, self.loss_fn = self.create_model()

    def set_params(self, model_params):
        self.model.set_weights(model_params)

    def get_params(self):
        return self.model.get_weights()

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A tuple consisting of:
                model: A Tensorflow model
                loss: A Tensorflow operation that, when run with features and labels,
                    returns the loss of the model.
        """
        return None, None

    def train(self, data):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
        Returns:
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        logging.debug("Training...")
        data_size = tf.constant(0, dtype=tf.int32)
        if self.prox_mu > 0:
            initial_weights = deepcopy(self.get_params())
        for x_batch, y_batch in data:
            data_size += tf.shape(x_batch)[0]
            with tf.GradientTape() as tape:
                logits = self.model(x_batch, training=True)
                loss_value = self.loss_fn(y_batch, logits)
                if self.prox_mu > 0:
                    #logging.debug(f'Proximal term: {tf.add_n([tf.nn.l2_loss(w1 - w2) for w1, w2 in zip(self.get_params(), initial_weights)])}')
                    #loss_value += self.prox_mu * tf.add_n([tf.nn.l2_loss(w1 - w2) for w1, w2 in zip(self.get_params(), initial_weights)])
                    loss_value += self.prox_mu / 2 * tf.norm(flatten(self.get_params()) - flatten(initial_weights))**2
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        logging.debug("Done Training.")
        update = self.get_params()
        return update

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Returns:
            dict of metrics that will be recorded by the simulation.
        """

        accuracy = tf.keras.metrics.Accuracy()
        avg_test_loss = tf.keras.metrics.Mean()
        total_conf_matrix = tf.zeros((self.num_classes, self.num_classes), dtype=tf.int32)
        for x_batch, y_batch in data:
            logits = self.model(x_batch, training=False)
            labels = tf.argmax(input=logits, axis=1)
            avg_test_loss.update_state(self.loss_fn(y_batch, logits))
            accuracy.update_state(y_batch, labels)
            total_conf_matrix += tf.math.confusion_matrix(y_batch, labels, num_classes=self.num_classes)

        return {'accuracy': accuracy.result().numpy(), 'conf_matrix': total_conf_matrix.numpy(),
                'loss': avg_test_loss.result().numpy(), 'additional_metrics': None}
