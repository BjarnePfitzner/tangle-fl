import itertools
import logging
import sys
import time

from omegaconf import DictConfig
import wandb

from data import get_dataset_instance
from models.model import Model
from tangle.lab import Lab
from tangle.progress import Progress


def main(cfg: DictConfig):
    # Get dataset
    dataset = get_dataset_instance(cfg.dataset, centralised=True, log_sample_data=cfg.log_sample_data)
    dataset.preprocess_datasets(cfg.run.batch_size, cfg.run.test_batch_size, None)

    # Get model
    model = Lab.create_client_model(cfg.seed, cfg.run, cfg.dataset)

    # Do training
    total_rounds_of_training = run_training(dataset, model, cfg)

    return total_rounds_of_training


def run_training(dataset, model: Model, cfg: DictConfig):
    if cfg.run.num_rounds == -1:
        rounds_iter = itertools.count(cfg.run.start_from_round)
        progress = Progress(1000, cfg.run.eval_every)
    else:
        rounds_iter = range(cfg.run.start_from_round, cfg.run.num_rounds)
        progress = Progress(cfg.run.num_rounds - cfg.run.start_from_round, cfg.run.eval_every)

    final_test_metrics = None
    for round in rounds_iter:
        begin = time.time()
        logging.info('Started training for round %s' % round)
        sys.stdout.flush()

        model.train(dataset.train_ds)
        val_metrics = model.test(dataset.val_ds)

        wandb.log({
            'val/loss': val_metrics['loss'],
            'val/accuracy': val_metrics['accuracy'],
        }, step=round)

        train_duration = time.time() - begin
        progress.add_train_duration(train_duration)
        wandb.log({'durations/train': train_duration}, step=round)

        test_duration = 0
        if cfg.run.eval_every != -1 and round % cfg.run.eval_every == 0 and round != cfg.run.num_rounds - 1:
            begin = time.time()
            test_metrics = model.test(dataset.test_ds)

            wandb.log({
                'test/loss': test_metrics['loss'],
                'test/accuracy': test_metrics['accuracy'],
            }, step=round)

            logging.info(f'Average accuracy: {test_metrics["accuracy"]}\nAverage loss: {test_metrics["loss"]}')

            if test_metrics['accuracy'] >= cfg.run.target_accuracy:
                logging.info(f'Stopping due to reaching of target accuracy ({test_metrics["accuracy"]} >= {cfg.run.target_accuracy})')
                wandb.log({'train/rounds_to_target_accuracy': round}, step=round)
                final_test_metrics = test_metrics
                break
            test_duration = time.time() - begin
            progress.add_eval_duration(test_duration)
            logging.info(f'Test duration: {test_duration:.2f}s')
            wandb.log({'durations/test': test_duration}, step=round)

        hours_left, mins_left = progress.eta(round)
        logging.info(f'This round took: {train_duration + test_duration:.2f}s - {str(hours_left) + "h " if hours_left > 0 else ""}{mins_left}m left')
        sys.stdout.flush()

    # Final evaluation
    if final_test_metrics is None:
        final_test_metrics = model.test(dataset.test_ds)
    wandb.log({
        'test/loss': final_test_metrics['loss'],
        'test/accuracy': final_test_metrics['accuracy'],
    }, step=round)

    wandb.log({'train/rounds_to_target_accuracy': round}, step=round)
    return round
