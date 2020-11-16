Config = {
    "TRAINING_INTERVAL": 25,
    "TIMEOUT": None,
    "NUM_OF_TIPS": 2,
    "NUM_SAMPLING_ROUND_CONFIDENCE": 35,
    "ACTIVE_QUOTA": 0.01
}


def set_config(args):
    global Config
    Config["TRAINING_INTERVAL"] = args.training_interval
    Config["TIMEOUT"] = args.timeout
    Config["NUM_OF_TIPS"] = args.NUM_OF_TIPS
    Config["NUM_SAMPLING_ROUND_CONFIDENCE"] = args.num_sampling_round
    Config['ACTIVE_QUOTA'] = args.active_quota

    return
