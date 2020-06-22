import os
import io
import hashlib
import numpy as np

from ..core import TransactionStore

class FilesystemTransactionStore:
    def __init__(self, tx_path):
        self.tx_path = tx_path

    def load_transaction_weights(self, tx):
        return np.load(f'{self.tx_path}/{tx.name()}.npy', allow_pickle=True)

    def compute_transaction_id(self, tx):
        tmpfile = io.BytesIO()
        self._save(tx, tmpfile)
        tmpfile.seek(0)
        return self.hash_file(tmpfile)

    def save(self, tx):
        os.makedirs(self.tx_path, exist_ok=True)

        with open(f'{self.tx_path}/{tx.name()}.npy', 'wb') as tx_file:
            self._save(tx, tx_file)

        tx.id = self.compute_transaction_id(tx)

    def _save(self, tx, file):
        np.save(file, tx.weights, allow_pickle=True)

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
