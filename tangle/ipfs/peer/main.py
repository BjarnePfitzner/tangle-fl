import asyncio
import argparse
import ipaddress
import logging
import random
import subprocess
import threading
import time
import importlib
from logging import FileHandler

import numpy as np

from . import logger
from .listener import Listener
from ...lab.dataset import read_data
from ...models.baseline_constants import MODEL_PARAMS
from .peer_http_server import PeerHttpServer
from .tangle import TangleBuilder
from .tangle.ipfs_transaction_store import IpfsTransactionStore
from ...core import Node, Transaction, Tangle

from ...lab.config import ModelConfiguration, NodeConfiguration, TipSelectorConfiguration
from ...lab.args import parse_args
from .config import PeerConfiguration
from .pubsub_client import PubsubClient
from .ipfs_client import IpfsClient


async def start_daemon():
    start = time.process_time()

    command = ['ipfs_entrypoint', 'daemon', '--migrate=true', '--enable-pubsub-experiment', '--enable-gc']
    d = subprocess.Popen(command, stdout=subprocess.PIPE)

    while d.stdout.readline() != b'Daemon is ready\n':
        pass

    calc_time = time.process_time() - start
    logger.info(f'IPFS daemon start: {calc_time}')

    ipfs_client = IpfsClient('http://127.0.0.1:5001')
    print(await ipfs_client.query_version())

    return d, ipfs_client

def create_genesis():
    # Required so tensorflow is not loaded by default
    import tensorflow as tf
    from .model import ClientModel
    tf.reset_default_graph()
    client_model = ClientModel(0)

    genesis_weights = client_model.get_params()
    with open(f'/data/genesis.npy', 'wb') as file:
        np.save(file, genesis_weights, allow_pickle=True)


def load_data():
    logger.info('Loading data...')
    train_clients, train_groups, train_data, test_data = read_data('/data/train', '/data/test')

    peer_ip = ipaddress.ip_address(subprocess.check_output(['hostname', '-i']).decode().strip())
    rnd = random.Random(4711)
    rnd.shuffle(train_clients)
    this_client = train_clients[int(peer_ip) % len(train_clients)]

    logger.info('Client ' + this_client)
    return this_client, train_data[this_client], test_data[this_client]

def parse_model(mock, model_config):
    if mock:
        from .model.no_tf_model import NoTfModel
        logger.info('No TF Model')
        return NoTfModel
    else:
        return create_client_model(random.randint(0, 4294967295), model_config)

def create_client_model(seed, model_config):
    model_path = '.%s.%s' % (model_config.dataset, model_config.model)
    mod = importlib.import_module(model_path, package='tangle.models')
    ClientModel = getattr(mod, 'ClientModel')

    # Create 2 models
    model_params = MODEL_PARAMS['%s.%s' % (model_config.dataset, model_config.model)]
    if model_config.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = model_config.lr
        model_params = tuple(model_params_list)

    model = ClientModel(seed, *model_params)
    model.num_epochs = model_config.num_epochs
    model.batch_size = model_config.batch_size
    model.num_batches = model_config.num_batches
    return model

async def load_genesis(genesis_path, tx_store):
    genesis_weights = np.load(genesis_path, allow_pickle=True)
    genesis = Transaction([])
    genesis.add_metadata('peer', {"client_id": "fffff_ff"})
    genesis.add_metadata('time', 0)
    await tx_store.save(genesis, genesis_weights)
    return genesis

async def main(loop):
    peer_config, model_config, node_config, tip_selector_config = \
        parse_args(PeerConfiguration, ModelConfiguration, NodeConfiguration, TipSelectorConfiguration)

    if peer_config.create_genesis:
        create_genesis()
        return

    client_id, train_data, test_data = load_data()
    model = parse_model(peer_config.mock_model, model_config)

    for h in logger.handlers:
        if type(h) is FileHandler:
            formatter = logging.Formatter(
                f'{client_id}| %(asctime)s | %(levelname)-6s | %(filename)s-%(funcName)s-%(lineno)04d | %(message)s')
            h.setFormatter(formatter)

    d, ipfs_client = await start_daemon()

    tx_store = IpfsTransactionStore(ipfs_client)
    genesis = await load_genesis('/data/genesis.npy', tx_store)
    tangle = Tangle({genesis.id: genesis}, genesis.id)
    message_broker = PubsubClient(ipfs_client, tangle.genesis)

    peer_information = {'client_id': client_id}
    tangle_builder = TangleBuilder(tangle, tx_store, message_broker, peer_information, train_data, test_data, tip_selector_config, model)
    threading.Thread(target=PeerHttpServer(tangle_builder.tangle, peer_information).start).start()
    listener = Listener(loop, tangle_builder, message_broker, train_data, test_data, peer_config)
    await listener.listen()

    # Stay alive
    d.communicate()
