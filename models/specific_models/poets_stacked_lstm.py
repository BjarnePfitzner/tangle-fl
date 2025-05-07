import tensorflow as tf

from tensorflow.keras import layers

from models.model import Model

class ClientModel(Model):
    def __init__(self, seed, lr, cfg, prox_mu=0):
        self.num_classes = cfg.num_classes
        self.seq_len = cfg.seq_len
        self.n_hidden = cfg.n_hidden
        super(ClientModel, self).__init__(seed, lr, prox_mu=prox_mu)

    def create_model(self):
        model = tf.keras.Sequential([
            layers.Embedding(input_dim=self.num_classes, output_dim=8, input_length=self.seq_len),
            layers.LSTM(self.n_hidden, return_sequences=True),
            layers.LSTM(self.n_hidden),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        return model, loss
