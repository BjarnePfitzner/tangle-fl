import random
import numpy as np

from ..core import Tangle, Transaction, Node, TransactionStore
from ..core.tip_selection import TipSelector

NUM_NODES = 1000
NUM_ROUNDS = 100
NODES_PER_ROUND = 10

DIST_STD_DEV = 100

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
        return { 'loss': abs(model_params - self.data )}

    def train(self, averaged_weights):
        diff = np.array(self.data) - np.array(averaged_weights)

        # Limit the 'learning rate' to 1
        length = np.linalg.norm(diff)
        if length > 1:
            step = diff / (length / 1)
        else:
            step = diff

        return averaged_weights + step

def main():
    tx_store = TempTransactionStore()

    genesis = Transaction([])
    tx_store.save(genesis, random.uniform(-DIST_STD_DEV, DIST_STD_DEV))

    tangle = Tangle({genesis.id: genesis}, genesis.id)

    mu, sigma = 0, DIST_STD_DEV # mean and standard deviation
    node_data = np.random.normal(mu, sigma, NUM_NODES)

    for r in range(NUM_ROUNDS):
        txs = []

        for n in range(NODES_PER_ROUND):
            node_id = np.random.randint(NUM_NODES)
            tip_selector = TipSelector(tangle)
            node = TheoreticalNode(tangle, tx_store, tip_selector, node_id, None, node_data[node_id])
            tx, tx_weights = node.create_transaction()

            if tx is not None:
                tx_store.save(tx, tx_weights)
                txs.append(tx)

        for tx in txs:
            tangle.add_transaction(tx)

        tip_selector = TipSelector(tangle)
        validation_node = TheoreticalNode(tangle, tx_store, tip_selector, 0, None, node_data[0])
        reference_txs, reference = node.obtain_reference_params()

        print('Round', r, 'consensus is', reference)
