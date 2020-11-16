import argparse
import ipaddress
import logging
import random
import subprocess
import threading
import time
from logging import FileHandler

import ipfshttpclient4ipwb
import numpy as np

from . import logger
from .config import Config, set_config
from .listener import Listener
from .model.utils.model_utils import read_data
from .peer_http_server import PeerHttpServer
from .tangle import TangleBuilder


def start_daemon():
    start = time.process_time()

    command = ["ipfs_entrypoint", "daemon", "--migrate=true", "--enable-pubsub-experiment", "--enable-gc"]
    d = subprocess.Popen(command, stdout=subprocess.PIPE)

    while d.stdout.readline() not in b'Daemon is ready\n':
        time.sleep(0.1)
    calc_time = time.process_time() - start
    logger.info(f'IPFS daemon start: {calc_time}')

    return d


def parse_args():
    def int_or_none(value):
        if isinstance(value, int) or value is None:
            return value
        elif isinstance(value, str) and value.isdigit():
            return int(value)
        elif value == "None":
            return None
        else:
            msg = f'{value} is neither int nore None'
            raise argparse.ArgumentTypeError(msg)

    parser = argparse.ArgumentParser()

    parser.add_argument('--create-genesis',
                        help='create a genesis transaction at /data/genesis.npy',
                        action='store_true')

    parser.add_argument('--storage', default='ipfs', help='sets the used storage')
    parser.add_argument('--broker', default='ipfs', help='sets the used message broker')
    parser.add_argument('--model', default='femnist', help='sets the used model')
    parser.add_argument('--timeout', default=None, type=int_or_none, help='timeout for ipfs')
    parser.add_argument('--training_interval', default=20, type=int, help='trainings interval')
    parser.add_argument('--num_of_tipps', default=2, type=int, help='number of tipps in tipp selection')
    parser.add_argument('--num_sampling_round', default=35, type=int, help='Number of transactions choosen for conses')
    parser.add_argument('--active_quota', type=float, default=0.01,
                        help='sets the quota of active pears, must be valid percentage (0.00, 0.01, ..., 0.99, 1.00')

    return parser.parse_args()


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


def get_ipfs_client():
    global __IPFS_CLIENT__
    while __IPFS_CLIENT__ == None:
        try:
            __IPFS_CLIENT__ = ipfshttpclient4ipwb.connect(session=True, timeout=Config["TIMEOUT"])
        except Exception as e:
            logger.error(e)
            __IPFS_CLIENT__ = None
    return __IPFS_CLIENT__


# Import only when needed


def parse_storage(storage):
    if storage == "ipfs":
        from .storage.ipfs_storage import IPFSStorage
        return IPFSStorage(get_ipfs_client())


# Import only when needed


def parse_message_broker(message_broker):
    if message_broker == "ipfs":
        from .message_broker.ipfs_message_broker import IPFSMessageBroker
        return IPFSMessageBroker(get_ipfs_client())


# Import only when needed


def parse_model(model):
    if model == "femnist":
        from .model.femnist import ClientModel
        return ClientModel
    elif model == "no_tf":
        from .model.no_tf_model import NoTfModel
        logger.info("No TF Model")
        return NoTfModel


async def main(loop):
    args = parse_args()
    if args.create_genesis:
        create_genesis()
        return
    set_config(args)
    for h in logger.handlers:
        if type(h) is FileHandler:
            formatter = logging.Formatter(
                f'{get_client()}| %(asctime)s | %(levelname)-6s | %(filename)s-%(funcName)s-%(lineno)04d | %(message)s')
            h.setFormatter(formatter)
    logger.info("Config: " + str(Config))
    d = start_daemon()

    train_data, test_data = load_data()
    message_broker = parse_message_broker(args.broker)
    storage = parse_storage(args.storage)
    model = parse_model(args.model)
    peer_information = {'client_id': get_client()}
    tangle_builder = TangleBuilder('/data/genesis.npy', message_broker, storage, model, peer_information)
    threading.Thread(target=PeerHttpServer(tangle_builder.tangle, peer_information).start).start()
    listener = Listener(loop, tangle_builder, message_broker, train_data, test_data)
    await listener.listen()
    # Stay alive
    d.communicate()
