"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import logging


class Model(ABC):
    num_classes: int

    def __init__(self, seed, lr, optimizer=None):
        self.lr = lr
        self.seed = seed
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
        for x_batch, y_batch in data:
            data_size += tf.shape(x_batch)[0]
            with tf.GradientTape() as tape:
                logits = self.model(x_batch, training=True)
                loss_value = self.loss_fn(y_batch, logits)
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
            total_conf_matrix += tf.math.confusion_matrix(labels, labels, num_classes=self.num_classes)

        return {'accuracy': accuracy.result().numpy(), 'conf_matrix': total_conf_matrix.numpy(),
                'loss': avg_test_loss.result().numpy(), 'additional_metrics': None}
