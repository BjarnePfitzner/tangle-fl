import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .args import parse_args

from . import Lab
from .config import ModelConfiguration, LabConfiguration, RunConfiguration

def main():
    run_config, lab_config, model_config = parse_args(RunConfiguration, LabConfiguration, ModelConfiguration)

    lab = Lab(lab_config, model_config)

    lab.train(run_config.clients_per_round, run_config.start_from_round, run_config.num_rounds)
    lab.validate(run_config.num_rounds-1)
