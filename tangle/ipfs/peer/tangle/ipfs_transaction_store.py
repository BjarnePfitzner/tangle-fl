import io
import numpy as np
from ....core import TransactionStore

class IPFSTransactionStore(TransactionStore):

    def __init__(self, ipfs_client):
        self._ipfs_client = ipfs_client
        self._tx_id_to_weights = {}

    # Are used to store & retrieve weights
    # Returns the file path or Hash
    def save(self, tx, tx_weights):
        # try:
        weights_bytes = io.BytesIO()
        np.save(weights_bytes, tx_weights, allow_pickle=True)
        weights_bytes.seek(0)
        weights_key = self.add_file(weights_bytes)

        tx.add_metadata('weights_ref', weights_key)

        tx.id = self.compute_transaction_id(tx)
        self._tx_id_to_weights[tx.id] = weights_key
        # except:
        #     pass

    def register_transaction(self, tx_id, tx_weight_id):
        self._tx_id_to_weights[tx_id] = tx_weight_id

    def load_transaction_weights(self, tx_id):
        if tx_id not in self._tx_id_to_weights:
            raise Exception('unknown transaction, please register first')

        try:
            file = self.get_file(self._tx_id_to_weights[tx_id])
            return np.load(io.BytesIO(file), allow_pickle=True)  # Potentially dangerous
        except:
            return None

    def compute_transaction_id(self, tx, only_hash=False):
        # TODO: It would be awesome if we could pass the only_hash parameter
        # on to the IPFS endpoint...
        return self.add_json({
            'parents': sorted(tx.parents),
            'weights': tx.metadata['weights_ref'],
            'peer': tx.metadata['peer']
        })

    # Are used to store & retrieve weights
    # Returns the file path or Hash
    def add_file(self, content):
        try:
            return self._ipfs_client.add(content)['Hash']
        except:
            return None

    # Returns the file content or Hash
    def get_file(self, path: str):
        try:
            return self._ipfs_client.cat(path)
        except:
            return None

    # Are used to store transactions and to retrieve missed transactions
    # Returns the file path or Hash
    def add_json(self, value: dict):
        try:
            return self._ipfs_client.add_json(value)
        except:
            return None

    def get_json(self, path: str):
        try:
            return self._ipfs_client.get_json(path)
        except:
            return None
