import tensorflow as tf

from ..model import Model
from ..utils.tf_utils import graph_size
from ..baseline_constants import ACCURACY_KEY
from ...lab.dataset import batch_data
import numpy as np


IMAGE_SIZE = 32
DROPOUT = 0.3


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, optimizer=None):
        self.lr = lr
        self.seed = seed
        self.num_classes = num_classes
        self._optimizer = optimizer

        self.num_epochs = 1
        self.batch_size = 10

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + self.seed)
            self.features, self.labels, self.dropout, self.train_op, self.eval_metric_ops, self.conf_matrix, self.loss, *self.additional_params = self.create_model()
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph,config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                        intra_op_parallelism_threads=1,
                                        use_per_session_threads=True))

        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        np.random.seed(self.seed)


    def create_model(self):
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        dropout = tf.placeholder(tf.float32, shape=None, name='dropout')

        # input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
        
        conv1 = tf.layers.conv2d(
          inputs=features,
          filters=64,
          kernel_size=[3, 3],
          kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.08),
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')
        bn1 = tf.layers.batch_normalization(pool1)
        
        conv2 = tf.layers.conv2d(
          inputs=bn1,
          filters=128,
          kernel_size=[3, 3],
          kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.08),
          padding="same",
          activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(pool2)

        conv3 = tf.layers.conv2d(
          inputs=bn2,
          filters=256,
          kernel_size=[5, 5],
          kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.08),
          padding="same",
          activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(pool3)

        conv4 = tf.layers.conv2d(
          inputs=bn3,
          filters=512,
          kernel_size=[5, 5],
          kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.08),
          padding="same",
          activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='same')
        bn4 = tf.layers.batch_normalization(pool4)

        bn4_flat = tf.layers.flatten(bn4)
        
        dense1 = tf.layers.dense(inputs=bn4_flat, units=128, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(dense1, dropout)
        bn5 = tf.layers.batch_normalization(dropout1)

        dense2 = tf.layers.dense(inputs=bn5, units=256, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(dense2, dropout)
        bn6 = tf.layers.batch_normalization(dropout2)

        dense3 = tf.layers.dense(inputs=bn6, units=512, activation=tf.nn.relu)
        dropout3 = tf.layers.dropout(dense3, dropout)
        bn7 = tf.layers.batch_normalization(dropout3)

        dense4 = tf.layers.dense(inputs=bn7, units=1024, activation=tf.nn.relu)
        dropout4 = tf.layers.dropout(dense4, dropout)
        bn8 = tf.layers.batch_normalization(dropout4)

        logits = tf.layers.dense(inputs=bn8, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        conf_matrix = tf.math.confusion_matrix(labels, predictions["classes"], num_classes=self.num_classes)
        return features, labels, dropout, train_op, eval_metric_ops, conf_matrix, loss

    # Overwrite for dropout input
    def run_epoch(self, data, batch_size):
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):

            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            with self.graph.as_default():
                self.sess.run(self.train_op,
                    feed_dict={
                        self.features: input_data,
                        self.labels: target_data,
                        self.dropout: DROPOUT
                    })

    # Overwrite for dropout input
    def test(self, data):
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        with self.graph.as_default():
            tot_acc, conf_matrix, loss, *adds = self.sess.run(
                [self.eval_metric_ops, self.conf_matrix, self.loss, *self.additional_params],
                feed_dict={self.features: x_vecs, self.labels: labels, self.dropout: 1.0}
            )
        acc = float(tot_acc) / x_vecs.shape[0]
        return {ACCURACY_KEY: acc, 'conf_matrix': conf_matrix, 'loss': loss, 'additional_metrics': adds}

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
