import datetime
import io

import numpy as np

from .. import logger
from ..message_broker.message_broker import MessageBroker
from ..metrics.counter_metrics import increment_count_transaction_published, increment_count_transaction_publish_error
from ..storage.storage import Storage


class Transaction:
    def __init__(self, tangle_id, weights_ref, parents, storage: Storage):
        self._tangle_id = tangle_id
        self._weights_ref = weights_ref
        self.parents = parents
        self.peer_information = {"client_id": "fffff_ff"}
        self.id = self.compute_id(storage)
        self.weights = None
        self.time = 0
        self.time_created = str(datetime.datetime.now())

    def compute_id(self, storage: Storage):
        return storage.add_json(self.envelope())

    def envelope(self):
        return {
            'parents': sorted(self.parents),
            'weights': self._weights_ref,
            'peer': self.peer_information
        }

    def load_weights(self, storage: Storage):
        try:
            bytes = storage.get_file(self._weights_ref)
            return np.load(io.BytesIO(bytes), allow_pickle=True)  # Potentially dangerous
        except:
            return None

    def publish(self, message_broker: MessageBroker, weights, peer_information):
        self.peer_information = peer_information
        envelope = self.envelope()
        publish_success = message_broker.publish(self._tangle_id, envelope)
        if publish_success:
            logger.info(f'Published transaction {self.id}')
            increment_count_transaction_published()
            return publish_success
        else:
            logger.warn(f'Failed to publish transaction {self.id}')
            increment_count_transaction_publish_error()
            return publish_success
