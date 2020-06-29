import argparse

DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']
SIM_TIMES = ['small', 'medium', 'large']
POISON_TYPES = ['none', 'random', 'labelflip']


def add_basic_args(parser):
    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    parser.add_argument('--metrics-name',
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--metrics-dir',
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--use-val-set',
                    help='use validation set;',
                    action='store_true')
    parser.add_argument('--model-data-dir',
                    help='dir for model data',
                    type=str,
                    default='data',
                    required=False)
    parser.add_argument('--tangle-dir',
                    help='dir for tangle data (DAG JSON)',
                    type=str,
                    default='tangle_data',
                    required=False)
    parser.add_argument('--tangle-tx-dir',
                    help='dir for tangle transaction data',
                    type=str,
                    default='transactions',
                    required=False)
    parser.add_argument('--start-from-round',
                    help='at which round to start/resume training',
                    type=int,
                    default=0)

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    epoch_capability_group.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)

def add_poisoning_args(parser):
    parser.add_argument('--poison-type',
                    help='type of malicious clients considered',
                    type=str,
                    choices=POISON_TYPES,
                    default='none',
                    required=False)
    parser.add_argument('--poison-fraction',
                    help='fraction of clients being malicious',
                    type=float,
                    default=0,
                    required=False)
    parser.add_argument('--poison-from',
                    help='epoch to start poisoning from',
                    type=float,
                    default=1,
                    required=False)

def add_tangle_hyperparameter_args(parser):
    parser.add_argument('--num-tips',
                    help='number of tips being selected per round',
                    type=int,
                    default=2,
                    required=False)
    parser.add_argument('--sample-size',
                    help='number of possible tips being sampled per round',
                    type=int,
                    default=2,
                    required=False)

    parser.add_argument('--target-accuracy',
                    help='stop training after reaching this test accuracy',
                    type=float,
                    default=1,
                    required=False)
    parser.add_argument('--reference-avg-top',
                    help='number models to average when picking reference model',
                    type=int,
                    default=1,
                    required=False)

def add_tip_selection_args(parser):
    # Parameters for AccuracyTipSelector
    parser.add_argument('--acc-tip-selection-strategy',
                    help='strategy how to select the next tips',
                    type=str,
                    choices=['WALK', 'GLOBAL'],
                    default='WALK')

    parser.add_argument('--acc-cumulate-ratings',
                    help='whether after calculating accuracies should be cumulated',
                    type=str2bool,
                    default=False)

    parser.add_argument('--acc-ratings-to-weights',
                    help='algorithm to generate weights from ratings. Has effect only if used with WALK',
                    type=str,
                    choices=['LINEAR'],
                    default='LINEAR')

    parser.add_argument('--acc-select-from-weights',
                    help='algorithm to select the next transaction from given weights. Has effect only if used with WALK',
                    type=str,
                    choices=['MAXIMUM', 'WEIGHTED_CHOICE'],
                    default='MAXIMUM')

    parser.add_argument('--acc-alpha',
                    help='exponential factor for ratings to weights. Has effect only if used with WALK and WEIGHTED_CHOICE',
                    type=float,
                    default=0.001)


def parse_args():
    parser = argparse.ArgumentParser()
    add_basic_args(parser)
    add_poisoning_args(parser)
    add_tangle_hyperparameter_args(parser)
    add_tip_selection_args(parser)

    result = parser.parse_args()

    # tip_selection_settings = {}
    # tip_selection_settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] = args.acc_tip_selection_strategy
    # tip_selection_settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS] = args.acc_cumulate_ratings
    # tip_selection_settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] = args.acc_ratings_to_weights
    # tip_selection_settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] = args.acc_select_from_weights
    # tip_selection_settings[AccuracyTipSelectorSettings.ALPHA] = args.acc_alpha

    return result

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
