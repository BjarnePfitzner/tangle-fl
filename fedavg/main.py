import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from omegaconf import DictConfig

from data import get_dataset_instance
from fedavg.fed_avg import FedAvg


def main(cfg: DictConfig):
    dataset = get_dataset_instance(cfg.dataset, cfg.log_sample_data)
    dataset.preprocess_datasets(cfg.run.batch_size, cfg.run.test_batch_size, cfg.run.local_epochs)

    fedavg = FedAvg(dataset, cfg)

    return fedavg.run_training()
