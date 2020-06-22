import numpy as np
import sys

from .transaction import Transaction

class NodeConfiguration:
    num_tips: int
    sample_size: int
    reference_avg_top: int

    def __init__(self, num_tips=2, sample_size=2, reference_avg_top=1):
        self.num_tips = num_tips
        self.sample_size = sample_size
        self.reference_avg_top = reference_avg_top

class Node:
    def __init__(self, tangle, tx_store, tip_selector, client_id, cluster_id, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self.tangle = tangle
        self.tx_store = tx_store
        self.tip_selector = tip_selector
        self._model = model
        self.id = client_id
        self.cluster_id = cluster_id
        self.train_data = train_data
        self.eval_data = eval_data
        self.config = NodeConfiguration()


    def train(self, num_epochs=1, batch_size=10):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.

        Return:
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        data = self.train_data
        update = self.model.train(data, num_epochs, batch_size)
        num_train_samples = len(data['y'])
        return num_train_samples, update

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.

        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def choose_tips(self, selector, num_tips=2, sample_size=2):
        if len(self.tangle.transactions) < num_tips:
            return [self.tangle.transactions[self.tangle.genesis] for i in range(2)]

        tips = selector.tip_selection(sample_size)

        no_dups = set(tips)
        if len(no_dups) >= num_tips:
            tips = no_dups

        tip_txs = [self.tangle.transactions[tip] for tip in tips]

        # Find best tips
        if num_tips < sample_size:
            # Choose tips with lowest test loss
            tip_losses = []
            loss_cache = {}
            for tip in tip_txs:
                if tip.id in loss_cache.keys():
                    tip_losses.append((tip, loss_cache[tip.id]))
                else:
                    self.model.set_params(self.tx_store.load_transaction_weights(tip))
                    loss = self.test('test')['loss']
                    tip_losses.append((tip, loss))
                    loss_cache[tip.id] = loss
            best_tips = sorted(tip_losses, key=lambda tup: tup[1], reverse=False)[:num_tips]
            tip_txs = [tup[0] for tup in best_tips]

        return tip_txs

    def compute_confidence(self, selector, approved_transactions_cache={}):
        num_sampling_rounds = 35

        transaction_confidence = {x: 0 for x in self.tangle.transactions}

        def approved_transactions(transaction):
            if transaction not in approved_transactions_cache:
                result = set([transaction]).union(*[approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
                approved_transactions_cache[transaction] = result

            return approved_transactions_cache[transaction]

        for i in range(num_sampling_rounds):
            tips = self.choose_tips(selector=selector)
            for tip in tips:
                for tx in approved_transactions(tip.name()):
                    transaction_confidence[tx] += 1

        return {tx: float(transaction_confidence[tx]) / (num_sampling_rounds * 2) for tx in self.tangle.transactions}

    def compute_cumulative_score(self, transactions, approved_transactions_cache={}):
        def compute_approved_transactions(transaction):
            if transaction not in approved_transactions_cache:
                result = set([transaction]).union(*[compute_approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
                approved_transactions_cache[transaction] = result

            return approved_transactions_cache[transaction]

        return {tx: len(compute_approved_transactions(tx)) for tx in transactions}

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

        # 3. For the top avg_top transactions, compute the average
        best = sorted(
            {tx: scores[tx] * transaction_confidence[tx] for tx in keys}.items(),
            key=lambda kv: kv[1], reverse=True
        )[:avg_top]
        reference_txs = [elem[0] for elem in best]
        reference_params = self.average_model_params(*[self.tx_store.load_transaction_weights(self.tangle.transactions[elem]) for elem in reference_txs])
        return reference_txs, reference_params

    def average_model_params(self, *params):
        return sum(params) / len(params)

    def create_transaction(self, num_epochs, batch_size):
        selector = self.tip_selector(self.tangle)

        # Compute reference metrics
        reference_txs, reference = self.obtain_reference_params(avg_top=self.config.reference_avg_top, selector=selector)
        self.model.set_params(reference)
        c_metrics = self.test('test')

        # Obtain number of tips from the tangle
        tips = self.choose_tips(selector=selector, num_tips=self.config.num_tips, sample_size=self.config.sample_size)

        # Perform averaging

        # How averaging is done exactly (e.g. weighted, using which weights) is left to the
        # network participants. It is not reproducible or verifiable by other nodes because
        # only the resulting weights are published.
        # Once a node has published its training results, it thus can't be sure if
        # and by what weight its delta is being incorporated into approving transactions.
        # However, assuming most nodes are well-behaved, they will make sure that eventually
        # those weights will prevail that incorporate as many partial results as possible
        # in order to prevent over-fitting.

        # Here: simple unweighted average
        averaged_weights = self.average_model_params(*[self.tx_store.load_transaction_weights(tip) for tip in tips])
        self.model.set_params(averaged_weights)
        num_samples, update = self.train(num_epochs, batch_size)

        c_averaged_model_metrics = self.test('test')
        if c_averaged_model_metrics['loss'] < c_metrics['loss']:
            return Transaction(self.model.get_params(), set([tip.name() for tip in tips]), self.id, self.cluster_id)

        return None
