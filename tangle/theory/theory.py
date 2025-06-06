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
        self.data = data
        super().__init__(tangle, tx_store, tip_selector, client_id, cluster_id)
        self.config.sample_size = 5

    def test(self, model_params, set_to_use='test', only_test_on_first_1000=False):
        error = np.linalg.norm(np.array(model_params) - np.array(self.data))
        return { 'loss': error, 'accuracy': 1-(error/250) }


    def train(self, model_params):
        diff = self.data - model_params

        # Limit the 'learning rate'
        max_step_length = 5
        length = np.linalg.norm(diff)
        if length > max_step_length:
            step = diff / (length / max_step_length)
        else:
            step = diff

        return model_params + step
