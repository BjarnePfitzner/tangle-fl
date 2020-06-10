#! /usr/bin/env python

import sys
import os
import itertools

from leaf.models.poison_type import PoisonType

sys.path.insert(1, './leaf/models')

import numpy as np
import tensorflow as tf

from tangle import Tangle, train_single, AccuracyTipSelectorSettings
from utils.model_utils import read_data
from utils.args import parse_args

def main():

    args = parse_args()

    # client_id = sys.argv[1]
    # tangle_name = sys.argv[2]
    client_id = 'f0044_12'
    cluster_id = '0'
    tangle_name = '10_clients_50'
    tip_selection_settings = { AccuracyTipSelectorSettings.SELECTION_STRATEGY: 'WALK',
                               AccuracyTipSelectorSettings.CUMULATE_RATINGS: False,
                               AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT: 'LINEAR',
                               AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS: 'WEIGHTED_CHOICE'}

    train_data_dir = os.path.join('leaf', 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join('leaf', 'data', args.dataset, 'data', 'test')

    print("Loading data...")
    users, cluster_ids, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    print("Loading data... complete")

    print(train_single(client_id, None, 1, 0, train_data[client_id], test_data[client_id], tangle_name, False, PoisonType.NONE,tip_selection_settings))

if __name__ == '__main__':
    main()
