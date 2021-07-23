import base64
import json

import aiohttp

from . import logger
from .metrics.counter_metrics import increment_counter_subscriber, increment_counter_subscriber_exit
from ...core import Transaction

class PubsubClient():

    def __init__(self, ipfs_client, genesis, tx_store):
        super().__init__()
        self._ipfsclient = ipfs_client
        self._tangle_id = genesis.id
        self._tx_store = tx_store

    async def subscribe(self):
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=None, connect=1, sock_connect=1, sock_read=None)
                    logger.info(f'Subscribing to tangle {self._tangle_id}')
                    # Abstraction 1 -> PubSub
                    async with session.post(f'http://127.0.0.1:5001/api/v0/pubsub/sub?arg={self._tangle_id}&arg=True',
                                            timeout=timeout) as resp:
                        increment_counter_subscriber()
                        logger.info(f'Subscribed to tangle {self._tangle_id}')
                        async for line in resp.content:
                            message = json.loads(line)
                            payload_bytes = base64.b64decode(message['data'])
                            tx_data = json.loads(payload_bytes)
                            tx = Transaction(tx_data['parents'])
                            tx.add_metadata('peer', tx_data['peer'])
                            tx.add_metadata('weights_ref', tx_data['weights'])
                            tx.add_metadata('time', 0)
                            tx.id = await self._tx_store.compute_transaction_id(tx, only_hash=True)
                            yield tx
            except Exception as e:
                logger.error("Tangle subscription died, session was teared down and will be restarted\n" + repr(e))

    async def publish(self, tx):
        envelope = {
            'parents': sorted(tx.parents),
            'weights': tx.metadata['weights_ref'],
            'peer': tx.metadata['peer']
        }
        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=None, connect=1, sock_connect=1, sock_read=None)
            async with session.post(f'http://127.0.0.1:5001/api/v0/pubsub/pub?arg={self._tangle_id}&arg={json.dumps(envelope)}',
                                    timeout=timeout) as response:
                if response.status != 200:
                    response_text = await response.text()
                    raise IpfsError(response_text)
