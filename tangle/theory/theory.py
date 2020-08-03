import numpy as np
from ..core import Node, TransactionStore

current_id = 0
def next_tx_id():
    global current_id
    id = current_id
    current_id = current_id + 1
    return id


class TempTransactionStore(TransactionStore):
    def __init__(self):
        self.weights = {}

    def load_transaction_weights(self, tx_id):
        return self.weights[tx_id]

    def compute_transaction_id(self, tx):
        return next_tx_id()

    def save(self, tx, tx_weights):
        tx.id = self.compute_transaction_id(tx_weights)
        self.weights[tx.id] = tx_weights


class TheoreticalNode(Node):
    def __init__(self, tangle, tx_store, tip_selector, client_id, cluster_id, data):
        super().__init__(tangle, tx_store, tip_selector, client_id, cluster_id)
        self.data = data

    def test(self, model_params, set_to_use='test'):
        return { 'loss': np.linalg.norm(np.array(model_params) - np.array(self.data)) }


    def train(self, averaged_weights):
        diff = np.array(self.data) - np.array(averaged_weights)

        # Limit the 'learning rate'
        max_step_length = 1
        length = np.linalg.norm(diff)
        if length > max_step_length:
            step = diff / (length / max_step_length)
        else:
            step = diff

        return averaged_weights + step
