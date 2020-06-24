import os
import io
import hashlib
import numpy as np

from ..core import TransactionStore

class FilesystemTransactionStore:
    def __init__(self, tangle_path, tx_path):
        self.tangle_path = tangle_path
        self.tx_path = tx_path

    def load_transaction_weights(self, tx_id):
        return np.load(f'{self.tx_path}/{tx_id}.npy', allow_pickle=True)

    def compute_transaction_id(self, tx_weights):
        tmpfile = io.BytesIO()
        self._save(tx_weights, tmpfile)
        tmpfile.seek(0)
        return self.hash_file(tmpfile)

    def save(self, tx, tx_weights):
        tx.id = self.compute_transaction_id(tx_weights)

        os.makedirs(self.tx_path, exist_ok=True)

        with open(f'{self.tx_path}/{tx.id}.npy', 'wb') as tx_file:
            self._save(tx_weights, tx_file)

    def _save(self, tx_weights, file):
        np.save(file, tx_weights, allow_pickle=True)

    @staticmethod
    def hash_file(f):
        BUF_SIZE = 65536
        sha1 = hashlib.sha1()
        while True:
          data = f.read(BUF_SIZE)
          if not data:
              break
          sha1.update(data)

        return sha1.hexdigest()
