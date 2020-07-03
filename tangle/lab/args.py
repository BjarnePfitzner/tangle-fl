import argparse

def parse_args(*config_cls):
    parser = argparse.ArgumentParser()

    configs = [c() for c in config_cls]

    for c in configs:
        c.define_args(parser)

    result = parser.parse_args()

    for c in configs:
        c.parse(result)

    # tip_selection_settings = {}
    # tip_selection_settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] = args.acc_tip_selection_strategy
    # tip_selection_settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS] = args.acc_cumulate_ratings
    # tip_selection_settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] = args.acc_ratings_to_weights
    # tip_selection_settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] = args.acc_select_from_weights
    # tip_selection_settings[AccuracyTipSelectorSettings.ALPHA] = args.acc_alpha

    return configs

