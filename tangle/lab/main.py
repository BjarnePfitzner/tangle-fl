import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .args import parse_args

from . import Lab, Dataset, TipSelectorFactory
from .config import LabConfiguration, ModelConfiguration, PoisoningConfiguration, RunConfiguration, TangleConfiguration, TipSelectorConfiguration

def main():
    run_config, lab_config, model_config, poisoining_config, tangle_config, tip_selector_config = \
        parse_args(RunConfiguration, LabConfiguration, ModelConfiguration, PoisoningConfiguration, TangleConfiguration, TipSelectorConfiguration)

    tip_selector_factory = TipSelectorFactory(tip_selector_config)
    lab = Lab(tip_selector_factory, lab_config, model_config)

    dataset = Dataset(lab_config, model_config)

    lab.train(run_config.clients_per_round, run_config.start_from_round, run_config.num_rounds, run_config.eval_every, dataset)
    print(lab.validate(run_config.num_rounds-1, dataset))
