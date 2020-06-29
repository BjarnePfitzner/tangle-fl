import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .args import parse_args

from .lab import ModelConfiguration, LabConfiguration, Lab

def main():
    args = parse_args()

    config = LabConfiguration(
        args.seed,
        args.model_data_dir,
        args.tangle_dir,
        args.tangle_tx_dir
    )

    model_config = ModelConfiguration(
        args.dataset,
        args.model,
        args.lr,
        args.use_val_set,
        args.num_epochs,
        args.batch_size
    )

    lab = Lab(config, model_config)

    lab.train(args.clients_per_round, args.start_from_round, args.num_rounds)
