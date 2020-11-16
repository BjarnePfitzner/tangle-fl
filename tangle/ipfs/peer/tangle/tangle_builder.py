import io
import os
import time
import datetime

import numpy as np

from ....core import Tangle, Transaction
from ....core.tip_selection import TipSelector
from .. import logger
from ..config import Config
from ..message_broker.message_broker import MessageBroker
from ..metrics.counter_metrics import *
from ..metrics.histogram_metrics import *

NUM_OF_TIPS = Config['NUM_OF_TIPS']
NUM_SAMPLING_ROUND_CONFIDENCE = Config['NUM_SAMPLING_ROUND_CONFIDENCE']

class TangleBuilder:
    def __init__(self, genesis_path, message_broker: MessageBroker, tx_store, model_type, peer_information):
        self.peer_information = peer_information
        self._message_broker = message_broker
        self._tx_store = tx_store
        self._model_type = model_type
        genesis = self._add_genesis(genesis_path)
        self.tangle = Tangle({genesis.id: genesis}, genesis.id)
        os.makedirs(f'iota_visualization/tangle_data', exist_ok=True)
        self.last_training = None
        # Hackedy-hack
        self._message_broker._tangle_id = genesis.id

    def _add_genesis(self, genesis_path):
        genesis_weights = np.load(genesis_path, allow_pickle=True)
        genesis = Transaction([])
        genesis.add_metadata('peer', {"client_id": "fffff_ff"})
        genesis.add_metadata('time', 0)
        self._tx_store.save(genesis, genesis_weights)
        return genesis

    def _choose_tips(self, selector=None):
        num_tips = NUM_OF_TIPS
        sample_size = NUM_OF_TIPS

        if selector is None:
            selector = TipSelector(self.tangle)
            selector.compute_ratings(None)

        if len(self.tangle.transactions) < num_tips:
            return [self.tangle.transactions[self.tangle.genesis] for _ in range(2)]

        tips = selector.tip_selection(sample_size)

        no_dups = set(tips)
        if len(no_dups) >= num_tips:
            tips = no_dups

        return [self.tangle.transactions[tip] for tip in tips]

    def _compute_confidence(self, selector=None, approved_transactions_cache={}):
        num_sampling_rounds = NUM_SAMPLING_ROUND_CONFIDENCE

        transaction_confidence = {x: 0 for x in self.tangle.transactions}

        def approved_transactions(transaction_id):

            if transaction_id not in approved_transactions_cache:
                queue = [transaction_id]
                current_id = transaction_id
                while len(queue) != 0:
                    p_queue = []
                    for p in self.tangle.transactions[current_id].parents:

                        if p not in approved_transactions_cache:
                            p_queue.append(p)
                    if len(p_queue) == 0:

                        result = {current_id}.union(
                            *[approved_transactions_cache[parent] for parent in self.tangle.transactions[current_id].parents])
                        approved_transactions_cache[current_id] = result
                    else:
                        queue.append(current_id)
                        queue.extend(p_queue)
                    current_id = queue.pop()

                result = {transaction_id}.union(
                    *[approved_transactions_cache[parent] for parent in self.tangle.transactions[transaction_id].parents])
                approved_transactions_cache[transaction_id] = result
            return approved_transactions_cache[transaction_id]

        # Use a cached tip selector
        if selector is None:
            selector = TipSelector(self.tangle)
            selector.compute_ratings(None)

        for i in range(num_sampling_rounds):
            tips = self._choose_tips(selector=selector)
            for tip in tips:
                for tx_id in approved_transactions(tip.id):
                    transaction_confidence[tx_id] += 1

        return {tx: float(transaction_confidence[tx]) / (num_sampling_rounds * 2) for tx in self.tangle.transactions}

    def _compute_cumulative_score(self, transactions, approved_transactions_cache={}):
        def compute_approved_transactions(transaction):
            if transaction not in approved_transactions_cache:
                result = {transaction}.union(*[compute_approved_transactions(parent) for parent in
                                               self.tangle.transactions[transaction].parents])
                approved_transactions_cache[transaction] = result

            return approved_transactions_cache[transaction]

        return {tx: len(compute_approved_transactions(tx)) for tx in transactions}

    @staticmethod
    def _average_model_params(params):
        return sum(params) / len(params)

    def _obtain_reference_params(self, selector=None):
        avg_top = 1
        # Establish the 'current best'/'reference' weights from the tangle

        approved_transactions_cache = {}

        # 1. Perform tip selection n times, establish confidence for each transaction
        # (i.e. which transactions were already approved by most of the current tips?)
        transaction_confidence = self._compute_confidence(selector=selector,
                                                          approved_transactions_cache=approved_transactions_cache)

        # 2. Compute cumulative score for transactions
        # (i.e. how many other transactions does a given transaction indirectly approve?)
        keys = [x for x in self.tangle.transactions]
        scores = self._compute_cumulative_score(keys, approved_transactions_cache=approved_transactions_cache)

        # 3. For the top 100 transactions, compute the average
        best = sorted(
            {tx: scores[tx] * transaction_confidence[tx] for tx in keys}.items(),
            key=lambda kv: kv[1], reverse=True
        )[:avg_top]
        reference_txs = [elem[0] for elem in best]

        filtered_reference_txs = self._filter_weights_loaded(reference_txs)
        if len(filtered_reference_txs) == 0:
            logger.info('References wrong')
            return None, None
        reference_params = self._average_model_params(
            [self._tx_store.load_transaction_weights(elem) for elem in filtered_reference_txs])
        return reference_txs, reference_params

    @staticmethod
    def _train(model, data, num_epochs=1, batch_size=10):
        '''Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        '''
        logger.info('Training...')
        update = model.train(data, num_epochs, batch_size)

        num_train_samples = len(data['y'])
        return num_train_samples, update

    def _filter_weights_loaded(self, tips):
        return [elem for elem in tips if self._tx_store.load_transaction_weights(elem) is not None]

    def train_and_publish(self, train_data, eval_data):
        start = time.time()
        if self.last_training is not None:
            logger.info(f'Time since last training: {start - self.last_training}')
            observe_time_between_training((start - self.last_training) / 1000)
        self.last_training = start

        self._train_and_publish(train_data, eval_data)
        train_duration = time.time() - start
        observe_time_training(train_duration / 1000)
        logger.info('Training duration in ms: ' + str(train_duration))

    def _train_and_publish(self, train_data, eval_data):
        logger.info('Start Training')

        increment_count_training()

        model = self._model_type(0)

        selector = TipSelector(self.tangle)
        selector.compute_ratings(None)

        # Compute reference metrics
        reference_txs, reference = self._obtain_reference_params(selector=selector)
        if reference_txs is None:
            increment_count_training_error()
            logger.info('Training Error: No Consens transactions available')
            return
        model.set_params(reference)
        c_metrics = model.test(eval_data)

        # Obtain number of tips from the tangle
        tips = self._choose_tips(selector=selector)

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

        selected_tips = [self._tx_store.load_transaction_weights(tip.id) for tip in tips
                          if self._tx_store.load_transaction_weights(tip.id) is not None]
        if len(selected_tips) == 0:
            logger.info('Training Error: Tip Selection Failed')
            return
        averaged_weights = self._average_model_params(selected_tips)
        model.set_params(averaged_weights)
        num_epochs = 1
        batch_size = 10
        _, _update = self._train(model, train_data, num_epochs, batch_size)

        c_averaged_model_metrics = model.test(eval_data)
        logger.info('Current model accuracy:' + str(c_averaged_model_metrics['accuracy']))
        observe_model_accuracy(float(c_averaged_model_metrics['accuracy']))
        logger.info('Current model loss:' + str(c_averaged_model_metrics['loss']))
        observe_model_loss(float(c_averaged_model_metrics['loss']))
        logger.info('Consensus model accuracy:' + str(c_metrics['accuracy']))

        observe_consensus_accuracy(float(c_metrics['accuracy']))
        logger.info('Consensus model loss:' + str(c_metrics['loss']))
        observe_consensus_model_loss(float(c_metrics['loss']))

        if c_averaged_model_metrics['loss'] < c_metrics['loss']:
            parents = set([tip.id for tip in tips])
            tx = Transaction(parents)
            tx.add_metadata('peer', self.peer_information)
            tx.add_metadata('time', 0)
            tx.add_metadata('timeCreated', str(datetime.datetime.now()))
            self._tx_store.save(tx, model.get_params())

            if tx.id is None:
                logger.warn('Publishing Error: Adding transactions failed')
                return

            publish_success = self._message_broker.publish(tx)
            if publish_success:
                logger.info(f'Published transaction {tx.id}')
                increment_count_transaction_published()
                return
            else:
                logger.warn(f'Failed to publish transaction {tx.id}')
                increment_count_transaction_publish_error()
                return
        else:
            increment_count_publish_error()
            logger.warn('Training Performance worse than consensus')

    def handle_transaction(self, tx_data):
        tx = Transaction(tx_data['parents'])
        tx.add_metadata('peer', tx_data['peer'])
        tx.add_metadata('weights_ref', tx_data['weights'])
        tx.add_metadata('time', 0)
        tx.id = self._tx_store.compute_transaction_id(tx, only_hash=True)

        self._tx_store.register_transaction(tx.id, tx_data['weights'])

        for parent in tx.parents:
            if parent in self.tangle.transactions:
                continue

            logger.info('Try to resolve tx' + str(parent))
            parent_tx_data = self._tx_store.get_json(parent)
            if parent_tx_data is not None:
                logger.info('Successfully resolved tx' + str(parent))
                self.handle_transaction(parent_tx_data)
            else:
                logger.warn('Parents not successfully resolved:' + str(parent))

        self.__order_transaction_time(tx)
        self.tangle.add_transaction(tx)
        increment_count_transaction_received()
        logger.info(f'Received transaction {tx.id}')

    def __order_transaction_time(self, tx):
        parents = tx.parents
        if len(parents) == 2:
            parent_time1 = self.tangle.transactions[parents[0]].metadata['time']
            parent_time2 = self.tangle.transactions[parents[1]].metadata['time']
            max_parent_time = max(parent_time1, parent_time2)
            time = max_parent_time + 0.1
        else:
            parent_time = self.tangle.transactions[parents[0]].metadata['time']
            time = parent_time + 0.1
        tx.add_metadata('time', time)

    # def close(self):
    #     self._client.close()
