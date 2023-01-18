import tensorflow as tf
from tensorflow.keras import layers

from ..model import Model
import numpy as np


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        model = tf.keras.models.Sequential(
            [
                layers.Conv2D(filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                              input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
                layers.MaxPool2D(pool_size=(2, 2), strides=2),
                layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
                layers.MaxPool2D(pool_size=(2, 2), strides=2),
                layers.Flatten(),
                layers.Dense(units=2048, activation=tf.nn.relu),
                layers.Dense(units=self.num_classes, activation=tf.nn.softmax)
            ]
        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return model, loss

    def process_x(self, raw_x_batch):
        return np.reshape(np.array(raw_x_batch), (-1, IMAGE_SIZE, IMAGE_SIZE, 1))

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
