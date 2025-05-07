import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from omegaconf import DictConfig

from data import get_dataset_instance
from tangle.lab import Lab
from tangle.tip_selector_factory import TipSelectorFactory


def main(cfg: DictConfig):
    tip_selector_factory = TipSelectorFactory(cfg.tip_selector)

    dataset = get_dataset_instance(cfg.dataset, log_sample_data=cfg.log_sample_data)
    dataset.preprocess_datasets(cfg.run.batch_size, cfg.run.test_batch_size, cfg.run.local_epochs)

    lab = Lab(tip_selector_factory, dataset, cfg)

    return lab.run_training()
