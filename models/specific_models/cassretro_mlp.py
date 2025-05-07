import tensorflow as tf
from tensorflow.keras import layers

from models.model import Model


class ClientModel(Model):
    def __init__(self, seed, lr, cfg):
        self.num_classes = cfg.num_classes
        self.data_dim = cfg.data_dim
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        model = tf.keras.models.Sequential(
            [
                layers.Dropout(0.2, input_shape=(self.data_dim,)),
                layers.Dense(units=64, activation=tf.nn.relu,
                             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
                layers.Dense(units=self.num_classes, activation=tf.nn.softmax,
                             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
            ]
        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return model, loss
