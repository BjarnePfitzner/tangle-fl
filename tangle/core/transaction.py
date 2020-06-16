import os
import hashlib
import io

from tempfile import TemporaryFile

import numpy as np

class Transaction:
    def __init__(self, weights, parents, client_id, cluster_id, id=None, tag=None, malicious=False, tangle_dir="."):
        self.weights = weights
        self.parents = parents
        self.client_id = client_id
        self.cluster_id = cluster_id
        self.tag = tag
        self.id = id
        self.malicious = malicious
        self.tangle_dir = tangle_dir

    def height(self, tangle):
      pass

    def load_weights(self, tangle_tx_dir):
        if self.weights is None and self.id is not None:
            self.weights = np.load(f'{tangle_tx_dir}/{self.id}.npy', allow_pickle=True)

        return self.weights

    def name(self):
        if self.id is None:
            tmpfile = io.BytesIO()
            self._save(tmpfile)
            tmpfile.seek(0)
            self.id = self.hash_file(tmpfile)

        return self.id

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

    def save(self, tangle_tx_dir):
        os.makedirs(tangle_tx_dir, exist_ok=True)

        with open(f'{tangle_tx_dir}/{self.name()}.npy', 'wb') as tx_file:
            self._save(tx_file)

    def _save(self, file):
      np.save(file, self.weights, allow_pickle=True)

    def add_tag(self, tag):
        self.tag = tag
