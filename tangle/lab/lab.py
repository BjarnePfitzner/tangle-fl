import random
import numpy as np
import importlib

from ..models.baseline_constants import MODEL_PARAMS
from ..core import Tangle, Transaction

class ModelConfiguration:
    dataset: str
    model: str
    lr: float

    def __init__(self, dataset, model, lr):
        self.dataset = dataset
        self.model = model
        self.lr = lr
        super().__init__()

class LabConfiguration:
    seed: int
    start_from_round: int
    tangle_dir: str
    tangle_tx_dir: str

    def __init__(self, seed, start_from_round, tangle_dir, tangle_tx_dir):
        self.seed = seed
        self.start_from_round = start_from_round
        self.tangle_dir = tangle_dir
        self.tangle_tx_dir = tangle_tx_dir
        super().__init__()


class Lab:
    def __init__(self, config, model_config):
        self.config = config
        self.model_config = model_config

        start_from_round = config.start_from_round

        # Set the random seed if provided (affects client sampling, and batching)
        random.seed(1 + config.seed)
        np.random.seed(12 + config.seed)

        if start_from_round == 0:
            genesis = self.create_genesis()

            tangle = Tangle({genesis: Transaction(None, [], None, None, genesis)}, genesis)
            tangle.save(config.tangle_dir, 0)
        else:
            tangle = Tangle.fromfile(config.tangle_data_dir, str(start_from_round))


    def create_genesis(self):
        import tensorflow as tf

        model_path = '.%s.%s' % (self.model_config.dataset, self.model_config.model)
        mod = importlib.import_module(model_path, package='tangle.models')
        ClientModel = getattr(mod, 'ClientModel')

        # Create 2 models
        model_params = MODEL_PARAMS['%s.%s' % (self.model_config.dataset, self.model_config.model)]
        if self.model_config.lr != -1:
            model_params_list = list(model_params)
            model_params_list[0] = self.model_config.lr
            model_params = tuple(model_params_list)

        client_model = ClientModel(self.config.seed, *model_params)

        genesis = Transaction(client_model.get_params(), [], "", None, tag=0)
        genesis.save(self.config.tangle_tx_dir)

        return genesis.name()
