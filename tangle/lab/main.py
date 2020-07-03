import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .args import parse_args

from . import Lab
from .config import LabConfiguration, ModelConfiguration, PoisoningConfiguration, RunConfiguration, TangleConfiguration, TipSelectorConfiguration

def main():
    run_config, lab_config, model_config, poisoining_config, tangle_config, tip_selector_config = \
        parse_args(RunConfiguration, LabConfiguration, ModelConfiguration, PoisoningConfiguration, TangleConfiguration, TipSelectorConfiguration)

    lab = Lab(lab_config, model_config, tip_selector_config)

    lab.train(run_config.clients_per_round, run_config.start_from_round, run_config.num_rounds)
    lab.validate(run_config.num_rounds-1)
