import string
import numpy as np

from data.abstract_dataset import AbstractDataset


class SyntheticDataset(AbstractDataset):
    @property
    def name(self):
        return 'Synthetic'

    @property
    def class_labels(self):
        return list(string.digits)

    @property
    def dataset_size(self):
        return {'train': int(np.sum(TOTAL_SIZES) * 0.9), 'test': int(np.sum(TOTAL_SIZES) * 0.1)}

    @property
    def avg_local_dataset_size(self):
        return np.mean(TOTAL_SIZES), np.std(TOTAL_SIZES)
        #return [int(s * 0.9) for s in TOTAL_SIZES], [int(s * 0.1) for s in TOTAL_SIZES]

    def get_dataset_size_for_client(self, client_id):
        assert not self.centralised
        return TOTAL_SIZES[np.argwhere(np.array(self.client_ids) == client_id)[0][0]]
        #train_sizes, test_sizes = [int(s * 0.9) for s in TOTAL_SIZES], [int(s * 0.1) for s in TOTAL_SIZES]
        #return train_sizes[np.argwhere(np.array(self.client_ids) == client_id)[0][0]], test_sizes[np.argwhere(np.array(self.client_ids) == client_id)[0][0]]

    def get_default_accuracy(self):
        return 0.989

TOTAL_SIZES = [89, 1320, 54, 99, 79, 165, 82, 841, 99, 59, 616, 153, 50, 647, 55, 54, 60, 52, 137, 92, 88, 74, 276, 1190, 75, 54, 104, 55, 55, 71]