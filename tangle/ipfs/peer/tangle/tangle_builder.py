import io
import os
import time
import datetime
import random

import numpy as np
import rx
import asyncio
from rx import operators as ops
from rx.subject import Subject
from rx.disposable import Disposable
from rx.scheduler.eventloop import AsyncIOScheduler

from ....core import Tangle, Transaction, Node
from ....core.tip_selection import TipSelector
from ....lab import TipSelectorFactory
from .. import logger
from ..metrics.counter_metrics import *
from ..metrics.histogram_metrics import *

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

class TangleBuilder:
    def __init__(self, loop, tx_store, message_broker, peer_information, train_data, test_data, tip_selector_config, model):
        self._loop = loop
        self.tangles = set()
        self.peer_information = peer_information
        self._tx_store = tx_store
        self._message_broker = message_broker
        self._train_data = train_data
        self._test_data = test_data
        self._tip_selector_config = tip_selector_config
        self._model = model
        self.current_tangle = None
        self.current_tips = None
        self.current_tx_weights = None
        self.reference_tx = None
        self.reference_tx_weights = None
        self.reference_tx_loss = None

        os.makedirs(f'iota_visualization/tangle_data', exist_ok=True)
        self.last_training = None

    async def initialize_reference_tx(self, tx):
        self.reference_tx = tx.id
        self.reference_tx_weights = await self._tx_store.load_transaction_weights(tx.id)
        node = Node(None, self._tx_store, None, self.peer_information['client_id'],
            None, self._train_data, self._test_data, self._model)
        metrics = node.test(self.reference_tx_weights, 'test')
        self.reference_tx_loss = metrics['loss']


    async def listen(self, genesis):
        done = asyncio.Future()
        scheduler = AsyncIOScheduler(self._loop)

        # TODO: If this peer was alive before, it should rather remember the best tx from last time
        await self.initialize_reference_tx(genesis)

        received_transactions = from_aiter(
            self._message_broker.subscribe(), self._loop)

        incoming_transactions = rx.just(genesis) \
            .pipe(ops.concat(received_transactions))

        # TODO: Resolve 'unresolved parents' and redirect them into 'incoming transactions'

        current_tip_updates = incoming_transactions \
            .pipe(ops.map(lambda x: self.update_tangles(x, genesis.id))) \
            .pipe(ops.throttle_first(datetime.timedelta(seconds=5), scheduler=scheduler)) \
            .pipe(ops.flat_map(lambda _: self.tip_selection())) \
            .pipe(ops.flat_map(lambda tangle_and_tips: self.resolve_weights(*tangle_and_tips))) \
            .pipe(ops.map(lambda tangle_and_tips_and_tx_weights: self.update_current_tips(*tangle_and_tips_and_tx_weights)))

        scheduled_trainings = rx.timer(0, datetime.timedelta(minutes=1), scheduler)
        training = current_tip_updates \
            .pipe(ops.merge(scheduled_trainings)) \
            .pipe(ops.filter(lambda _: self.current_tangle is not None)) \
            .pipe(ops.throttle_first(datetime.timedelta(minutes=1), scheduler=scheduler)) \
            .pipe(ops.flat_map(lambda _: self.train())) \
            .pipe(ops.flat_map(lambda tx_and_tx_weights: self.publish(*tx_and_tx_weights)))

        with training.subscribe(
                on_completed=lambda: done.set_result(),
                on_error=lambda e: done.set_exception(e),
                scheduler=scheduler):
            await done

    def update_tangles(self, tx, genesis):
        logger.info(f'Importing transaction {tx.id}')
        self._tx_store.register_transaction(tx.id, tx.metadata['weights_ref'])

        merge_tangles = set()

        for tangle in self.tangles:
            if tx.id in tangle.transactions.keys():
                assert len(merge_tangles) == 0
                return 1

            if tx.id in tangle.unresolved_parents:
                tangle.add_transaction(tx)
                merge_tangles.add(tangle)

            for parent in tx.parents:
                if parent in tangle.transactions:
                    tangle.add_transaction(tx)
                    merge_tangles.add(tangle)

        if len(merge_tangles) == 1:
            # Nothing more to do
            pass
        elif len(merge_tangles) > 1:
            # Merge the tangles
            target_tangle = merge_tangles.pop()
            for other_tangle in merge_tangles:
                target_tangle.unresolved_parents.update(other_tangle.unresolved_parents)

                for tx in other_tangle.transactions.values():
                    target_tangle.add_transaction(tx)

                self.tangles.remove(other_tangle)
        else:
            # Create a new tangle
            g = None
            unresolved_parents = []
            if tx.id == genesis:
                g = genesis
            else:
                unresolved_parents = [tx.id]
            tangle = Tangle({tx.id: tx}, g, unresolved_parents=unresolved_parents)
            self.tangles.add(tangle)

        return 1

    def tip_selection(self):
        return rx.from_future(self._loop.create_task(self._tip_selection()))

    async def _tip_selection(self):
        #  Choose tangle
        # TODO: Improve selection criteria
        tangle = random.choice(list(self.tangles))

        # Choose pair of tips
        trunk, branch = tangle.choose_trunk_branch()
        tip_selector = TipSelector(tangle, trunk=trunk, branch=branch)
        node = Node(tangle, self._tx_store, tip_selector, self.peer_information['client_id'],
            None, self._train_data, self._test_data, self._model)

        tips = await node.choose_tips()
        return tangle, tips

    def resolve_weights(self, tangle, tips):
        return rx.from_future(self._loop.create_task(self._resolve_weights(tangle, tips)))

    async def _resolve_weights(self, tangle, tips):
        tx_weights = [await self._tx_store.load_transaction_weights(tip.id) for tip in tips]
        return tangle, tips, tx_weights

    def update_current_tips(self, tangle, tips, tx_weights):
        self.current_tangle = tangle
        self.current_tips = tips
        self.current_tx_weights = tx_weights
        return 1

    def train(self):
        return rx.from_future(self._loop.create_task(self._train()))

    async def _train(self):
        # trunk, branch = self.current_tangle.choose_trunk_branch()
        # tip_selector = TipSelector(self.current_tangle, trunk=trunk, branch=branch)

        node = Node(self.current_tangle, self._tx_store, None, self.peer_information['client_id'],
                        None, self._train_data, self._test_data, self._model)

        tx, tx_weights = await node._create_transaction(self.current_tips, self.current_tx_weights, reference_tx=self.reference_tx, reference_tx_weights=self.reference_tx_weights)
        return tx, tx_weights

    def publish(self, tx, tx_weights):
        return rx.from_future(self._loop.create_task(self._publish(tx, tx_weights)))

    async def _publish(self, tx, tx_weights):
        if tx is not None:
            tx.add_metadata('peer', self.peer_information)
            tx.add_metadata('time', 0)
            tx.add_metadata('timeCreated', str(datetime.datetime.now()))
            await self._tx_store.save(tx, tx_weights)

            self.reference_tx = tx.id
            self.reference_tx_weights = tx_weights
            self.reference_tx_loss = tx.metadata['loss']

            if tx.id is None:
                logger.warn('Publishing Error: Adding transactions failed')
            else:
                try:
                    await self._message_broker.publish(tx)
                    logger.info(f'Published transaction {tx.id}')
                except:
                    logger.warn(f'Failed to publish transaction {tx.id}')
        return 1

    # async def train_and_publish(self, train_data, eval_data):
    #     start = time.time()
    #     if self.last_training is not None:
    #         logger.info(f'Time since last training: {start - self.last_training}')
    #         observe_time_between_training((start - self.last_training) / 1000)
    #     self.last_training = start

    #     tip_selector_factory = TipSelectorFactory(self._tip_selector_config)
    #     tip_selector = tip_selector_factory.create(self.tangle)
    #     node = Node(self.tangle, self._tx_store, tip_selector, self.peer_information['client_id'],
    #         None, self._train_data, self._test_data, self._model)

    #     logger.info('Start Training')
    #     # try:
    #     tx, tx_weights = await node.create_transaction()
    #     if tx is not None:
    #         tx.add_metadata('peer', self.peer_information)
    #         tx.add_metadata('time', 0)
    #         tx.add_metadata('timeCreated', str(datetime.datetime.now()))
    #         await self._tx_store.save(tx, tx_weights)

    #         if tx.id is None:
    #             logger.warn('Publishing Error: Adding transactions failed')
    #         else:
    #             # try:
    #             await self._message_broker.publish(tx)
    #             logger.info(f'Published transaction {tx.id}')
    #             increment_count_transaction_published()
    #             # except:
    #             #     logger.warn(f'Failed to publish transaction {tx.id}')
    #             #     increment_count_transaction_publish_error()
    #     else:
    #         increment_count_publish_error()
    #         logger.warn('Training Performance worse than consensus')

    #     train_duration = time.time() - start
    #     observe_time_training(train_duration / 1000)
    #     logger.info('Training duration in ms: ' + str(train_duration))
    #     # except:
    #     #     logger.info('Training Error')
    #     #     increment_count_training_error()

    # def __order_transaction_time(self, tx):
    #     parents = tx.parents
    #     if len(parents) == 2:
    #         parent_time1 = self.tangle.transactions[parents[0]].metadata['time']
    #         parent_time2 = self.tangle.transactions[parents[1]].metadata['time']
    #         max_parent_time = max(parent_time1, parent_time2)
    #         time = max_parent_time + 0.1
    #     else:
    #         parent_time = self.tangle.transactions[parents[0]].metadata['time']
    #         time = parent_time + 0.1
    #     tx.add_metadata('time', time)
