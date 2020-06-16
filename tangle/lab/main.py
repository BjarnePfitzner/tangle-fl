from .args import parse_args

from .lab import ModelConfiguration, LabConfiguration, Lab

def main():
    args = parse_args()

    config = LabConfiguration(
        args.seed,
        args.start_from_round,
        args.tangle_dir,
        args.tangle_tx_dir
    )

    model_config = ModelConfiguration(
        args.dataset,
        args.model,
        args.lr
    )

    lab = Lab(config, model_config)
