import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray
import wandb

from ..lab import parse_args
from ..lab.config import LabConfiguration, ModelConfiguration, RunConfiguration, NodeConfiguration, TipSelectorConfiguration, PoisoningConfiguration, WandBConfiguration

from . import RayDataset, RayTipSelectorFactory, RayLab

def main():
    run_config, lab_config, model_config, node_config, tip_selector_config, poisoning_config, wandb_config = parse_args(
        RunConfiguration, LabConfiguration, ModelConfiguration, NodeConfiguration, TipSelectorConfiguration,
        PoisoningConfiguration, WandBConfiguration)

    wandb_tags = [model_config.dataset, tip_selector_config.tip_selector]
    if  poisoning_config.poison_type != 'disabled':
        wandb_tags.append(f'{poisoning_config.poison_type}_poisoning')
    complete_config = {
        'run': run_config.__dict__,
        'lab': lab_config.__dict__,
        'model': model_config.__dict__,
        'tip_selector': tip_selector_config.__dict__,
        'poisoning': poisoning_config.__dict__
    }
    wandb.init(project="Tangle", entity="bjarnepfitzner",
               group=wandb_config.group, name=wandb_config.name, tags=wandb_tags, id=wandb_config.run_id, resume='allow',
               config=complete_config, allow_val_change=True,
               #settings=wandb.Settings(start_method="fork")
               )

    ray.init(webui_host='0.0.0.0')

    dataset = RayDataset(lab_config, model_config)

    tip_selector_factory = RayTipSelectorFactory(tip_selector_config)
    lab = RayLab(tip_selector_factory, lab_config, model_config, node_config, poisoning_config)

    lab.train(run_config.clients_per_round, run_config.start_from_round, run_config.num_rounds, run_config.eval_every, run_config.eval_on_fraction, dataset)
    #lab.print_validation_results(lab.validate(run_config.num_rounds-1, dataset, run_config.eval_on_fraction), mode='all')
