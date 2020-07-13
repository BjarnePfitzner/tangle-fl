import numpy as np
import sys

from .tip_selection import TipSelector
from .tip_selection import MaliciousTipSelector
from .transaction import Transaction
from .poisoning.poison_type import PoisonType

class MaliciousNode(Node):
    def __init__(self, tangle, client_id, cluster_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None, poison_type):
        self.poison_type = poison_type
        super().__init__(tangle, MaliciousTipSelector, client_id, cluster_id, group, train_data, eval_data, model)

    def choose_tips(self, selector, num_tips=2, sample_size=2):
        return super().choose_tips(selector, num_tips, num_tips)

    def compute_poisoning_score(self, transactions, approved_transactions_cache={}):
        def compute_approved_transactions(transaction):
            if transaction not in approved_transactions_cache:
                result = set([transaction]).union(*[compute_approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
                approved_transactions_cache[transaction] = result

            return approved_transactions_cache[transaction]

        return {tx: int(self.tangle.transactions[tx].malicious) + sum([self.tangle.transactions[transaction].malicious for transaction in compute_approved_transactions(tx)]) for tx in transactions}

    def obtain_reference_params(self, selector, avg_top=1):
        # Establish the 'current best'/'reference' weights from the tangle

        approved_transactions_cache = {}

        # 1. Perform tip selection n times, establish confidence for each transaction
        # (i.e. which transactions were already approved by most of the current tips?)
        transaction_confidence = self.compute_confidence(selector=selector, approved_transactions_cache=approved_transactions_cache)

        # 2. Compute cumulative score for transactions
        # (i.e. how many other transactions does a given transaction indirectly approve?)
        keys = [x for x in self.tangle.transactions]
        scores = self.compute_cumulative_score(keys, approved_transactions_cache=approved_transactions_cache)

        # How many directly or indirectly approved transactions are poisonous
        poison_scores = self.compute_poisoning_score(keys, approved_transactions_cache=approved_transactions_cache)
        poison_percentages = {tx: poison_scores[tx]/scores[tx] for tx in keys}

        # 3. For the top 100 transactions, compute the average
        best = sorted(
            {tx: scores[tx] * transaction_confidence[tx] for tx in keys}.items(),
            key=lambda kv: kv[1], reverse=True
        )[:avg_top]
        reference_txs = [elem[0] for elem in best]
        reference_params = self.average_model_params(*[self.tangle.transactions[elem].load_weights() for elem in reference_txs])
        reference_poison_score = np.mean([poison_percentages[elem] for elem in reference_txs])
        return reference_txs, reference_params, reference_poison_score

    def process_next_batch(self, num_epochs, batch_size, num_tips=2, sample_size=2, reference_avg_top=1, tip_selection_settings={}):
        selector = MaliciousTipSelector(self.tangle, self.client, tip_selection_settings)

        # Obtain number of tips from the tangle
        tips = self.choose_tips(selector=selector, num_tips=num_tips, sample_size=sample_size)

        if self.poison_type == PoisonType.RANDOM:
            weights = self.client.model.get_params()
            malicious_weights = [np.random.RandomState().normal(size=w.shape) for w in weights]
            print('generated malicious weights')
            return Transaction(malicious_weights, set([tip.id for tip in tips]), self.client.id, self.client.cluster_id, malicious=True), None, None
        elif self.poison_type == PoisonType.LABELFLIP:
            # Todo Choose tips or reference model?
            averaged_weights = self.average_model_params(*[tip.load_weights() for tip in tips])
            self.client.model.set_params(averaged_weights)
            self.client.train(num_epochs, batch_size)
            print('trained on label-flip data')
            return Transaction(self.client.model.get_params(), set([tip.id for tip in tips]), self.client.id, self.client.cluster_id, malicious=True), None, None

        return None, None, None
