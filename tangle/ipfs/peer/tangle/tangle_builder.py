import io
import os
import time
import datetime

import numpy as np

from ....core import Tangle, Transaction
from ....core.tip_selection import TipSelector
from .. import logger
from ..message_broker.message_broker import MessageBroker
from ..metrics.counter_metrics import *
from ..metrics.histogram_metrics import *

class TangleBuilder:
    def __init__(self, tangle, tx_store, message_broker: MessageBroker, node):
        self.tangle = tangle
        self.peer_information = {'client_id': node.id}
        self._tx_store = tx_store
        self._message_broker = message_broker
        self._node = node
        os.makedirs(f'iota_visualization/tangle_data', exist_ok=True)
        self.last_training = None

    # TODO: port this? was used filtering results of tip selection at computing consensus and training
    def _filter_weights_loaded(self, tips):
        return [elem for elem in tips if self._tx_store.load_transaction_weights(elem) is not None]

    def train_and_publish(self, train_data, eval_data):
        start = time.time()
        if self.last_training is not None:
            logger.info(f'Time since last training: {start - self.last_training}')
            observe_time_between_training((start - self.last_training) / 1000)
        self.last_training = start

        logger.info('Start Training')
        try:
            tx, tx_weights = self._node.create_transaction()
            if tx is not None:
                tx.add_metadata('peer', self.peer_information)
                tx.add_metadata('time', 0)
                tx.add_metadata('timeCreated', str(datetime.datetime.now()))
                self._tx_store.save(tx, tx_weights)

                if tx.id is None:
                    logger.warn('Publishing Error: Adding transactions failed')
                else:
                    publish_success = self._message_broker.publish(tx)
                    if publish_success:
                        logger.info(f'Published transaction {tx.id}')
                        increment_count_transaction_published()
                    else:
                        logger.warn(f'Failed to publish transaction {tx.id}')
                        increment_count_transaction_publish_error()
            else:
                increment_count_publish_error()
                logger.warn('Training Performance worse than consensus')

            train_duration = time.time() - start
            observe_time_training(train_duration / 1000)
            logger.info('Training duration in ms: ' + str(train_duration))
        except:
            logger.info('Training Error')
            increment_count_training_error()

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
