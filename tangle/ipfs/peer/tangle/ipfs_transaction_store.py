import io
import numpy as np
from ....core import TransactionStore, Transaction

class IpfsTransactionStore(TransactionStore):

    def __init__(self, ipfs_client):
        self._ipfs_client = ipfs_client
        self._tx_id_to_weights = {}

    # Are used to store & retrieve weights
    # Returns the file path or Hash
    async def save(self, tx, tx_weights):
        weights_bytes = io.BytesIO()
        np.save(weights_bytes, tx_weights, allow_pickle=True)
        weights_bytes.seek(0)
        weights_key = await self.add_file(weights_bytes)

        tx.add_metadata('weights_ref', weights_key)

        tx.id = await self.compute_transaction_id(tx)
        self._tx_id_to_weights[tx.id] = weights_key

    async def load(self, tx_id):
        tx_data = await self.get_json(tx_id)

        tx = Transaction(tx_data['parents'])
        tx.add_metadata('peer', tx_data['peer'])
        tx.add_metadata('weights_ref', tx_data['weights'])
        tx.add_metadata('time', 0)
        tx.id = tx_id

        return tx

    def register_transaction(self, tx_id, tx_weight_id):
        self._tx_id_to_weights[tx_id] = tx_weight_id

    async def load_transaction_weights(self, tx_id):
        if tx_id not in self._tx_id_to_weights:
            raise Exception('unknown transaction, please register first')

        file = await self._ipfs_client.cat(self._tx_id_to_weights[tx_id])
        return np.load(io.BytesIO(file), allow_pickle=True)  # Potentially dangerous

    async def compute_transaction_id(self, tx, only_hash=False):
        r = await self._ipfs_client.add_json({
            'parents': sorted(tx.parents),
            'weights': tx.metadata['weights_ref'],
            'peer': tx.metadata['peer']
        }, only_hash=True)
        return r['Hash']

    # Are used to store & retrieve weights
    # Returns the file path or Hash
    async def add_file(self, content):
        r = await self._ipfs_client.add_bytes(content)
        return r['Hash']

    async def get_json(self, path: str):
        return await self._ipfs_client.get_json(path)
