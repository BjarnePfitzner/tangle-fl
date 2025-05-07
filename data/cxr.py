import os
from pathlib import Path
import math

import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from data.abstract_dataset import AbstractDataset


path = Path(os.getenv('CHEST_XRAY_PATH', '/dhc/dsets/chest_xray_pneumonia_kaggle'))


class CXRDataset(AbstractDataset):
    @property
    def class_labels(self):
        return ['Normal', 'Pneumonia']

    @property
    def dataset_size(self):
        # https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
        return {'train': 5232, 'test': 656}

    @property
    def avg_local_dataset_size(self):
        return 436, 0

    def get_default_accuracy(self):
        return 0.748

    def get_dataset_size_for_client(self, client_id):
        return 436

    def _load_tf_dataset(self):
        def read_img(file_path):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img,
                                  size=[self.cfg.data_dim, self.cfg.data_dim],
                                  method='bilinear', antialias=True)
            if self.normalisation_mean_zero:
                img = (img * 2) - 1

            if tf.strings.split(file_path, os.path.sep)[-2] == 'NORMAL':
                label = 0
            else:
                label = 1

            return img, label

        train_data = tf.data.Dataset.list_files(f'{path}/train/*/*.jpeg', f'{path}/val/*/*.jpeg')
        test_data = tf.data.Dataset.list_files(f'{path}/test/*/*.jpeg')

        complete_train_ds = (train_data
                             .shuffle(buffer_size=self.dataset_size['train'], reshuffle_each_iteration=False)
                             .map(read_img, num_parallel_calls=AUTOTUNE))
        self.train_ds = (complete_train_ds
                         .take(math.floor(self.dataset_size['train'] * (1 - self.cfg.val_fraction)))
                         .cache())
        self.val_ds = (complete_train_ds
                       .skip(math.floor(self.dataset_size['train'] * (1 - self.cfg.val_fraction)))
                       .cache())
        self.test_ds = (test_data
                        .shuffle(buffer_size=self.dataset_size['test'], reshuffle_each_iteration=False)
                        .map(read_img, num_parallel_calls=AUTOTUNE)
                        .cache())
