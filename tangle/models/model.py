"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

from .baseline_constants import ACCURACY_KEY

from ..lab.dataset import batch_data


class Model(ABC):
    num_batches: int
    num_classes: int

    def __init__(self, seed, lr, optimizer=None):
        self.lr = lr
        self.seed = seed
        self.batch_seed = 12 + seed
        self._optimizer = optimizer

        self.num_epochs = 1
        self.batch_size = 10

        tf.random.set_seed(123 + self.seed)
        np.random.seed(self.seed)

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
        print("Training...")
        for _ in range(self.num_epochs):
            self.run_epoch(data, self.batch_size, self.num_batches)
        print("Done Training.")
        update = self.get_params()
        return update

    def run_epoch(self, data, batch_size, num_batches):
        self.batch_seed += 1

        for batched_x, batched_y in batch_data(data, batch_size, num_batches, seed=self.batch_seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            with tf.GradientTape() as tape:
                logits = self.model(input_data, training=True)
                loss_value = self.loss_fn(target_data, logits)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Returns:
            dict of metrics that will be recorded by the simulation.
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        # print(f"testing on {len(labels)} data")

        val_logits = self.model(x_vecs, training=False)
        val_labels = tf.argmax(input=val_logits, axis=1)
        val_loss_value = self.loss_fn(labels, val_logits)
        accuracy = float(tf.math.count_nonzero(tf.equal(labels, val_labels))) / x_vecs.shape[0]
        conf_matrix = tf.math.confusion_matrix(labels, val_labels, num_classes=self.num_classes)
        adds = None

        return {ACCURACY_KEY: accuracy, 'conf_matrix': conf_matrix, 'loss': val_loss_value, 'additional_metrics': adds}

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass


# class ServerModel:
#     def __init__(self, model):
#         self.model = model
#
#     @property
#     def size(self):
#         return self.model.size
#
#     @property
#     def cur_model(self):
#         return self.model
#
#     def send_to(self, clients):
#         """Copies server model variables to each of the given clients
#
#         Args:
#             clients: list of Client objects
#         """
#         var_vals = {}
#         with self.model.graph.as_default():
#             all_vars = tf.trainable_variables()
#             for v in all_vars:
#                 val = self.model.sess.run(v)
#                 var_vals[v.name] = val
#         for c in clients:
#             with c.model.graph.as_default():
#                 all_vars = tf.trainable_variables()
#                 for v in all_vars:
#                     v.load(var_vals[v.name], c.model.sess)
#
#     def save(self, path='checkpoints/model.ckpt'):
#         return self.model.saver.save(self.model.sess, path)
#
#     def close(self):
#         self.model.close()
