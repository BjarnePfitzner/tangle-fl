#! /usr/bin/env python

import itertools
import os
import sys
from shutil import copy, rmtree
from distutils.dir_util import copy_tree

from leaf.models.poison_type import PoisonType

sys.path.insert(1, './leaf/models')

import numpy as np
import tensorflow as tf

from tangle import Tangle, train_single, AccuracyTipSelectorSettings
from utils.model_utils import read_data
from utils.args import parse_args

def main():

    args = parse_args()

    ###### Parameters ######
    experiment_name = 'synthetic-log_reg-0'
    config = 0
    last_generation = 5     # The generation from where to start
    num_clients = 3         # The number of clients used per round

    client_id = '96'
    cluster_id = '2'        # Arbitrary value, as it has no effect on the calculation, nor will it be in the output
    ########################

    tangle_name = '%s_clients_%s' % (num_clients, last_generation)
    tangle_path = os.path.join('experiments', experiment_name, 'config_%s' % config, 'tangle_data')

    tip_selection_settings = { AccuracyTipSelectorSettings.SELECTION_STRATEGY: 'WALK',
                               AccuracyTipSelectorSettings.CUMULATE_RATINGS: False,
                               AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT: 'LINEAR',
                               AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS: 'WEIGHTED_CHOICE',
                               AccuracyTipSelectorSettings.ALPHA: 0.001 }

    train_data_dir = os.path.join('leaf', 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join('leaf', 'data', args.dataset, 'data', 'test')

    print("Loading data...")
    users, cluster_ids, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    print("Loading data... complete")

    # Copy tangle data from experiments folder, because:
    #   * working directory needs to be "learning-tangle", because relative import paths are used
    #   * it is not possible to specify the tangle file path directly, because it will be prefixed with a hardcoded string (tangle.py)
    print("Copying data...")
    os.makedirs('tangle_data', exist_ok=True)

    # Copy tangle file
    tangle_file = os.path.join(tangle_path, 'tangle_%s.json' % tangle_name)
    copy(tangle_file, 'tangle_data')

    # Copy transaction data
    tangle_transactions = os.path.join(tangle_path, 'transactions')
    copy_tree(tangle_transactions, os.path.join('tangle_data', 'transactions'))
    print("Copying data... completed")

    # Perform the step
    print(train_single(client_id, cluster_id, last_generation + 1, 0, 0, train_data[client_id], test_data[client_id], tangle_name, False, PoisonType.NONE, tip_selection_settings))

    # Delete copied data
    rmtree('tangle_data')

if __name__ == '__main__':
    main()
