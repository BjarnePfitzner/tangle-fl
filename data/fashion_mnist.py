import tensorflow as tf

from data.abstract_dataset import AbstractDataset


class FashionMNISTDataset(AbstractDataset):
    @property
    def name(self):
        return 'FashionMNIST'

    @property
    def class_labels(self):
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    @property
    def dataset_size(self):
        if self.cfg.use_val_data:
            return {'train': 60000, 'test': 5000, 'val': 5000}
        return {'train': 60000, 'test': 10000}

    def get_default_accuracy(self):
        return 0.989    # todo

    def _load_tf_dataset(self):
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
        if self.normalisation_mean_zero:
            train_x = train_x / 127.5 - 1
            test_x = test_x / 127.5 - 1
        else:
            train_x = train_x / 255
            test_x = test_x / 255

        train_x = tf.cast(train_x, tf.float32)
        test_x = tf.cast(test_x, tf.float32)

        self.train_ds = (tf.data.Dataset.from_tensor_slices((tf.expand_dims(train_x, -1), train_y))
                         .shuffle(buffer_size=self.dataset_size['train'], reshuffle_each_iteration=False)
                         .cache())
        self.test_ds = (tf.data.Dataset.from_tensor_slices((tf.expand_dims(test_x, -1), test_y))
                        .shuffle(buffer_size=self.dataset_size['test'], reshuffle_each_iteration=False))
        if self.cfg.use_val_data:
            self.val_ds = self.test_ds.shard(num_shards=2, index=0).cache()
            self.test_ds = self.test_ds.shard(num_shards=2, index=1).cache()
