import tensorflow as tf
from tensorflow.keras import layers

from models.model import Model


class ClientModel(Model):
    def __init__(self, seed, lr, cfg):
        self.num_classes = cfg.num_classes
        self.data_dim = cfg.data_dim
        self.data_ch = cfg.data_ch
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        model = tf.keras.models.Sequential(
            [
                layers.Conv2D(filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                              input_shape=(self.data_dim, self.data_dim, self.data_ch),
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
                layers.MaxPool2D(pool_size=(2, 2), strides=2),

                layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
                layers.MaxPool2D(pool_size=(2, 2), strides=2),

                layers.Conv2D(filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
                layers.MaxPool2D(pool_size=(2, 2), strides=2),

                layers.Flatten(),

                layers.Dense(units=256, activation=tf.nn.relu,
                             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
                layers.Dense(units=128, activation=tf.nn.relu,
                             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
                layers.Dense(units=self.num_classes, activation=tf.nn.softmax,
                             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
            ]
        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return model, loss
