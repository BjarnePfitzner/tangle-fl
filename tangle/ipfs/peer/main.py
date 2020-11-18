import argparse
import ipaddress
import logging
import random
import subprocess
import threading
import time
import importlib
from logging import FileHandler

import ipfshttpclient4ipwb
import numpy as np

from . import logger
from .listener import Listener
from ...lab.dataset import read_data
from ...models.baseline_constants import MODEL_PARAMS
from .peer_http_server import PeerHttpServer
from .tangle import TangleBuilder
from ...core import Node, Transaction, Tangle

from ...lab.config import ModelConfiguration, TangleConfiguration, TipSelectorConfiguration
from ...lab.args import parse_args
from .config import PeerConfiguration


def start_daemon():
    start = time.process_time()

    command = ['ipfs_entrypoint', 'daemon', '--migrate=true', '--enable-pubsub-experiment', '--enable-gc']
    d = subprocess.Popen(command, stdout=subprocess.PIPE)

    while d.stdout.readline() not in b'Daemon is ready\n':
        time.sleep(0.1)
    calc_time = time.process_time() - start
    logger.info(f'IPFS daemon start: {calc_time}')

    return d

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
    this_client = get_client()
    logger.info('Client ' + this_client)
    return train_data[this_client], test_data[this_client]


def get_client():
    train_clients, train_groups, train_data, test_data = read_data('/data/train', '/data/test')
    peer_ip = ipaddress.ip_address(subprocess.check_output(['hostname', '-i']).decode().strip())
    rnd = random.Random(4711)
    rnd.shuffle(train_clients)
    this_client = train_clients[int(peer_ip) % len(train_clients)]
    return this_client


__IPFS_CLIENT__ = None


# Only creates ipfs client when needed


def get_ipfs_client(timeout):
    global __IPFS_CLIENT__
    while __IPFS_CLIENT__ == None:
        try:
            __IPFS_CLIENT__ = ipfshttpclient4ipwb.connect(session=True, timeout=timeout)
        except Exception as e:
            logger.error(e)
            __IPFS_CLIENT__ = None
    return __IPFS_CLIENT__

def parse_storage(storage, timeout):
    if storage == 'ipfs':
        from .tangle.ipfs_transaction_store import IPFSTransactionStore
        return IPFSTransactionStore(get_ipfs_client(timeout))

def parse_message_broker(message_broker, timeout, genesis):
    if message_broker == 'ipfs':
        from .message_broker.ipfs_message_broker import IPFSMessageBroker
        return IPFSMessageBroker(get_ipfs_client(timeout), genesis)

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
    return model

def load_genesis(genesis_path, tx_store):
    genesis_weights = np.load(genesis_path, allow_pickle=True)
    genesis = Transaction([])
    genesis.add_metadata('peer', {"client_id": "fffff_ff"})
    genesis.add_metadata('time', 0)
    tx_store.save(genesis, genesis_weights)
    return genesis

async def main(loop):
    peer_config, model_config, tangle_config, tip_selector_config = \
        parse_args(PeerConfiguration, ModelConfiguration, TangleConfiguration, TipSelectorConfiguration)

    if peer_config.create_genesis:
        create_genesis()
        return

    for h in logger.handlers:
        if type(h) is FileHandler:
            formatter = logging.Formatter(
                f'{get_client()}| %(asctime)s | %(levelname)-6s | %(filename)s-%(funcName)s-%(lineno)04d | %(message)s')
            h.setFormatter(formatter)

    d = start_daemon()

    train_data, test_data = load_data()
    model = parse_model(peer_config.mock_model, model_config)

    tx_store = parse_storage(peer_config.storage, peer_config.timeout)
    genesis = load_genesis('/data/genesis.npy', tx_store)
    tangle = Tangle({genesis.id: genesis}, genesis.id)
    message_broker = parse_message_broker(peer_config.broker, peer_config.timeout, tangle.genesis)

    peer_information = {'client_id': get_client()}
    tangle_builder = TangleBuilder(tangle, tx_store, message_broker, peer_information, train_data, test_data, tip_selector_config, model)
    threading.Thread(target=PeerHttpServer(tangle_builder.tangle, peer_information).start).start()
    listener = Listener(loop, tangle_builder, message_broker, train_data, test_data, peer_config)
    await listener.listen()

    # Stay alive
    d.communicate()
