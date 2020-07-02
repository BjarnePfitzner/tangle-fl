import ray

from ..lab import LabTransactionStore

class RayTransactionStore(LabTransactionStore):
    def __init__(self, tangle_path, tx_path):
        super().__init__(tangle_path, tx_path)
        self.tx_cache = {}

    def load_transaction_weights(self, tx_id):
        if tx_id in self.tx_cache:
            return ray.get(self.tx_cache[tx_id])

        return super().load_transaction_weights(tx_id)

    def save(self, tx, tx_weights):
        super().save(tx, tx_weights)
        self.tx_cache[tx.id] = ray.put(tx_weights)
