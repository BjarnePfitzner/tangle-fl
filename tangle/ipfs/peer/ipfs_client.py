import base64
import json

import aiohttp

from . import logger
from .metrics.counter_metrics import increment_counter_subscriber, increment_counter_subscriber_exit
class IpfsError(Exception):
    pass

class IpfsClient():
    def __init__(self, url, timeout=10):
        self._url = url
        self._timeout = timeout

    async def query_version(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self._url}/api/v0/version') as response:
                if response.status == 200:
                    return await response.json()
                response_text = await response.text()
                raise IpfsError(response_text)

    async def add_bytes(self, bytes, timeout=None):
        async with aiohttp.ClientSession() as session:
            # timeout = aiohttp.ClientTimeout(total=None, connect=1, sock_connect=1, sock_read=None)
            with aiohttp.MultipartWriter('form-data') as writer:
                writer.append(bytes)
                async with session.post(f'{self._url}/api/v0/add', data=writer) as response:
                    if response.status == 200:
                        return await response.json()
                    response_text = await response.text()
                    raise IpfsError(response_text)

    async def add_json(self, value, only_hash=False):
        async with aiohttp.ClientSession() as session:
            # timeout = aiohttp.ClientTimeout(total=None, connect=1, sock_connect=1, sock_read=None)
            with aiohttp.MultipartWriter('form-data') as writer:
                writer.append_json(value)
                url = f'{self._url}/api/v0/add'
                if only_hash:
                    url += '?only_hash=true'
                async with session.post(url, data=writer) as response:
                    if response.status == 200:
                        return await response.json()
                    response_text = await response.text()
                    raise IpfsError(response_text)


    async def cat(self, cid):
        async with aiohttp.ClientSession() as session:
            # timeout = aiohttp.ClientTimeout(total=None, connect=1, sock_connect=1, sock_read=None)
            async with session.post(f'{self._url}/api/v0/cat?arg={cid}') as response:
                if response.status == 200:
                    return await response.read()
                response_text = await response.text()
                raise IpfsError(response_text)

    async def get_json(self, cid):
        async with aiohttp.ClientSession() as session:
            # timeout = aiohttp.ClientTimeout(total=None, connect=1, sock_connect=1, sock_read=None)
            async with session.post(f'{self._url}/api/v0/cat?arg={cid}') as response:
                if response.status == 200:
                    return await response.json(content_type='text/plain')
                response_text = await response.text()
                raise IpfsError(response_text)
