import argparse

class TipSelectorConfiguration:

    def define_args(self, parser):
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

    def parse_args(self, args):
        self.acc_tip_selection_strategy = args.acc_tip_selection_strategy
        self.acc_cumulate_ratings = args.acc_cumulate_ratings
        self.acc_ratings_to_weights = args.acc_ratings_to_weights
        self.acc_select_from_weights = args.acc_select_from_weights
        self.acc_alpha = args.acc_alpha

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
