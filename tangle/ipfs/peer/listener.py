import asyncio
import functools
import random

import rx
from rx import operators as ops
from rx.disposable import Disposable
from rx.scheduler.eventloop import AsyncIOScheduler

from . import logger

# https://blog.oakbits.com/rxpy-and-asyncio.html

class Event:
    def __init__(self, type, transaction=None):
        self.type = type
        self.transaction = transaction


def from_aiter(iter, loop):
    def on_subscribe(observer, scheduler):
        async def _aio_sub():
            try:
                async for i in iter:
                    observer.on_next(i)
                loop.call_soon(
                    observer.on_completed)
            except Exception as e:
                loop.call_soon(
                    functools.partial(observer.on_error, e))

        task = asyncio.ensure_future(_aio_sub(), loop=loop)
        return Disposable(lambda: task.cancel())

    return rx.create(on_subscribe)


class Listener:
    def __init__(self, loop, tangle_builder, message_broker, train_data, test_data, config):
        self._loop = loop
        self._tangle_builder = tangle_builder
        self._train_data = train_data
        self._test_data = test_data
        self._ready = False
        self._message_broker = message_broker
        self._config = config

    async def listen(self, genesis):
        done = asyncio.Future()

        received_transactions = from_aiter(
            self._message_broker.subscribe(self.on_ready), self._loop)

        with rx.just(genesis) \
            .pipe(ops.concat(received_transactions)) \
            .pipe(ops.map(lambda x: self.dispatch(x))) \
            .subscribe(
                on_completed=lambda: done.set_result(),
                on_error=lambda e: done.set_exception(e)):
            await done

    def dispatch(self, tx):
        self._tangle_builder.handle_transaction(tx)

    def on_ready(self):
        self._ready = True
