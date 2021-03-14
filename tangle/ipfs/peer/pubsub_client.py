import base64
import json

import aiohttp

from . import logger
from .metrics.counter_metrics import increment_counter_subscriber, increment_counter_subscriber_exit


class Event:
    def __init__(self, type, transaction=None):
        self.type = type
        self.transaction = transaction


class PubsubClient():

    def __init__(self, ipfs_client, genesis):
        super().__init__()
        self._ipfsclient = ipfs_client
        self._tangle_id = genesis

    async def subscribe(self, on_ready=None):
        retry_counter = 10
        while retry_counter > 0:
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=None, connect=1, sock_connect=1, sock_read=None)
                    # Abstraction 1 -> PubSub
                    async with session.post(f'http://127.0.0.1:5001/api/v0/pubsub/sub?arg={self._tangle_id}&arg=True',
                                            timeout=timeout) as resp:
                        increment_counter_subscriber()
                        logger.info(f'Subscribed to tangle {self._tangle_id}')
                        if on_ready:
                            on_ready()
                        async for line in resp.content:
                            message = json.loads(line)
                            payload_bytes = base64.b64decode(message['data'])
                            yield Event('transaction', json.loads(payload_bytes))
            except Exception as e:
                retry_counter = retry_counter - 1
                logger.error("Tangle subscription died, session was teared down and will be restarted another " + str(
                        retry_counter) + " times\n" + repr(e))
                if retry_counter == 0:
                    increment_counter_subscriber_exit()

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
