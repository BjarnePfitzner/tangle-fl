import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import datetime

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

import tangle.main as tangle_main
import centralised.main as centralised_main
import fedavg.main as fedavg_main
from tangle.analysis import TangleAnalyser


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.debug:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    cfg.experiment_folder = prepare_exp_folder(cfg)

    # If num_tips > sample_size, set sample_size to num_tips to allow the correct number of tips
    if cfg.node.num_tips > cfg.node.sample_size:
        cfg.node.sample_size = cfg.node.num_tips
    # If clients per round is absolute number, convert to fraction
    if cfg.run.clients_per_round > 1:
        cfg.run.clients_per_round = cfg.run.clients_per_round / cfg.dataset.num_clients

    # =================================
    # ========== Setup WandB ==========
    # =================================
    if cfg.wandb.disabled:
        logging.info('Disabled Wandb logging')
        wandb.init(mode='disabled')
    else:
        if cfg.resume_trial is None:
            wandb_id = wandb.util.generate_id()
            logging.info(f'New trial with id {wandb_id}')
        else:
            wandb_id = cfg.resume_trial
            logging.info(f'Resuming trial with id {wandb_id}')
            #cfg.run.start_from_round =

        wandb_tags = [cfg.dataset.name]
        if cfg.experiment_type == 'tangle':
            wandb_tags.append(cfg.tip_selector.type)
        else:
            wandb_tags.append(cfg.experiment_type)
        if cfg.poisoning.type != 'disabled':
            wandb_tags.append(f'{cfg.poisoning.type}_poisoning')

        wandb.init(project=cfg.wandb.project, entity="bjarnepfitzner",
                   group=cfg.wandb.group, name=cfg.wandb.name, tags=wandb_tags, id=wandb_id,
                   resume='allow', config=OmegaConf.to_container(cfg, resolve=True), allow_val_change=True,
                   dir=cfg.experiment_folder, settings=wandb.Settings(start_method="thread")
                   )

    config_filename = f'{cfg.experiment_folder}/hydra_config.yaml'
    if cfg.resume_trial is not None:
        # ========== Load config file ==========
        with open(config_filename, 'r') as f:
            cfg = OmegaConf.load(f)
    else:
        # ========== Save config file ==========
        yaml = OmegaConf.to_yaml(cfg)
        logging.info(yaml)
        with open(config_filename, 'w') as f:
            f.write(yaml)
    start_time = datetime.datetime.now()

    try:
        if cfg.experiment_type == 'tangle':
            total_rounds_of_training = tangle_main.main(cfg)
        elif cfg.experiment_type == 'centralised':
            total_rounds_of_training = centralised_main.main(cfg)
        elif cfg.experiment_type in ['fedavg', 'fedprox']:
            total_rounds_of_training = fedavg_main.main(cfg)

        # Document end of training
        logging.info(f'Training finished after {total_rounds_of_training} rounds.')

        end_time = datetime.datetime.now()
        logging.info('StartTime: %s' % start_time)
        logging.info('EndTime: %s' % end_time)
        logging.info('Duration Training: %s' % (end_time - start_time))

        if cfg.run_tangle_analyser and cfg.experiment_type == 'tangle':
            print('Analysing tangle...')
            analysis_folder = cfg.experiment_folder + '/tangle_analysis'
            os.makedirs(analysis_folder, exist_ok=True)
            analysator = TangleAnalyser(f'{cfg.experiment_folder}/{cfg.tangle_dir}', total_rounds_of_training,
                                        analysis_folder)
            analysator.save_statistics(include_reference_statistics=(cfg.node.publish_if_better_than == 'REFERENCE'),
                                       include_cluster_statistics=cfg.dataset.clustering,
                                       include_poisoning_statistics=(cfg.poisoning.type != 'disabled'))
            wandb.save(f'{analysis_folder}/*')
    finally:
        # Clean up
        if cfg.experiment_type == 'tangle' and (cfg.clean_up.transactions or cfg.clean_up.tangle_data):
            import shutil
            cleanup_path = f'{cfg.experiment_folder}/{cfg.tangle_dir}/'
            if not OmegaConf.is_missing(HydraConfig.get().job, "num"):
                cleanup_path = HydraConfig.get().sweep.dir + HydraConfig.get().sweep.subdir + '/' + cleanup_path[2:]

            if cfg.clean_up.transactions:
                # Delete .npy files with models
                logging.info(f'Cleaning up large transaction files in {to_absolute_path(cleanup_path)}')
                shutil.rmtree(to_absolute_path(cleanup_path + 'transactions'), ignore_errors=True)

            if cfg.clean_up.tangle_data:
                # Delete .json files of tangle data
                all_tangle_files = next(os.walk(cleanup_path))[2]
                largest_round_number = max([int(f.split('_')[-1].split('.')[0]) for f in all_tangle_files])
                [os.remove(cleanup_path + f) for f in all_tangle_files if int(f.split('_')[-1].split('.')[0]) < largest_round_number]


def prepare_exp_folder(cfg):
    experiments_base = './experiments'
    os.makedirs(experiments_base, exist_ok=True)

    prefix = cfg.wandb.name or cfg.experiment_name or f'{cfg.dataset.name}-{cfg.run.model}'

    # Find other experiments with default names
    all_experiments = next(os.walk(experiments_base))[1]
    default_exps = [exp for exp in all_experiments if re.match("^(%s-\d+)$" % prefix, exp)]

    # Find the last experiments with default name and increment id
    if len(default_exps) == 0:
        next_default_exp_id = 0
    else:
        default_exp_ids = [int(exp.split("-")[-1]) for exp in default_exps]
        default_exp_ids.sort()
        next_default_exp_id = default_exp_ids[-1] + 1

    if cfg.wandb.name:
        # treat named wandb runs differently
        if len(default_exps) == 0:
            cfg.experiment_name = cfg.wandb.name
        else:
            cfg.experiment_name = "%s-%d" % (prefix, len(default_exps) + 1)
    else:
        cfg.experiment_name = "%s-%d" % (prefix, next_default_exp_id)

    experiment_folder = experiments_base + '/' + cfg.experiment_name
    os.makedirs(experiment_folder, exist_ok=True)

    return experiment_folder


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
