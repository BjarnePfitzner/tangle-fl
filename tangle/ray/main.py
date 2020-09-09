import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray

from ..lab import parse_args
from ..lab.config import LabConfiguration, ModelConfiguration, PoisoningConfiguration, RunConfiguration, TangleConfiguration, TipSelectorConfiguration

from . import RayDataset, RayTipSelectorFactory, RayLab

def main():
    run_config, lab_config, model_config, poisoning_config, tangle_config, tip_selector_config = \
        parse_args(RunConfiguration, LabConfiguration, ModelConfiguration, PoisoningConfiguration, TangleConfiguration, TipSelectorConfiguration)

    ray.init(webui_host='0.0.0.0')

    dataset = RayDataset(lab_config, model_config)

    tip_selector_factory = RayTipSelectorFactory(tip_selector_config)
    lab = RayLab(tip_selector_factory, lab_config, model_config)

    lab.train(run_config.clients_per_round, run_config.start_from_round, run_config.num_rounds, run_config.eval_every, run_config.eval_on_fraction, dataset)
    #lab.print_validation_results(lab.validate(run_config.num_rounds-1, dataset, run_config.eval_on_fraction), mode='all')