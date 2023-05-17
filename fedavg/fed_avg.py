import sys
import time
import itertools
import logging
import random

import wandb
from omegaconf import DictConfig
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from zlib import crc32

from data.abstract_dataset import AbstractDataset
from tangle.lab import Lab
from tangle.core import Node, MaliciousNode, PoisonType
from tangle.progress import Progress


class DummyTipSelector():
    def compute_ratings(self, node):
        pass


class FedAvg:
    def __init__(self, dataset: AbstractDataset, cfg: DictConfig):
        self.dataset = dataset
        self.active_clients_list = self.dataset.client_ids[:]
        self.cfg = cfg

        # Set the random seed if provided (affects client sampling, and batching)
        random.seed(1 + cfg.seed)
        np.random.seed(12 + cfg.seed)

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        model = Lab.create_client_model(self.cfg.seed, self.cfg.run, self.cfg.dataset)
        self.global_params = model.get_params()

    def run_training(self):
        if self.cfg.run.num_rounds == -1:
            rounds_iter = itertools.count(self.cfg.run.start_from_round)
            progress = Progress(1000, self.cfg.run.eval_every)
        else:
            rounds_iter = range(self.cfg.run.start_from_round, self.cfg.run.num_rounds)
            progress = Progress(self.cfg.run.num_rounds - self.cfg.run.start_from_round, self.cfg.run.eval_every)

        all_publishing_clients = set([])
        total_client_contributions = 0
        final_test_metrics = None

        global_lr = 1.0
        for round in rounds_iter:
            begin = time.time()
            logging.info('Started training for round %s' % round)
            sys.stdout.flush()

            all_updates = self.train_one_round(round)

            param_update = 0
            total_weight = 0
            all_metrics = []
            for param_diff, weight, metrics in all_updates:
                param_update += param_diff * weight
                total_weight += weight
                total_client_contributions += 1
                all_metrics.append(metrics)

            param_update /= total_weight

            self.global_params -= global_lr * param_update

            all_publishing_clients = all_publishing_clients.union(
                set([client_metrics['client_id'] for client_metrics in all_metrics]))

            # Log some metrics
            wandb.log({
                'train/total_number_publishing_clients': len(all_publishing_clients),
                'val/loss': wandb.Histogram([client_metrics['val_loss'] for client_metrics in all_metrics]),
                'val/loss_avg': np.mean([client_metrics['val_loss'] for client_metrics in all_metrics]),
                'val/accuracy': wandb.Histogram([client_metrics['val_accuracy'] for client_metrics in all_metrics]),
                'val/accuracy_avg': np.mean([client_metrics['val_accuracy'] for client_metrics in all_metrics]),
                'information_gain/parent_txs_avg': np.mean(
                    [client_metrics['val_accuracy'] - client_metrics['old_val_accuracy'] for client_metrics in all_metrics]),
            }, step=round)
            # if self.cfg.poisoning.type != 'disabled':
            #     # todo add poisoning metrics
            #     wandb.log({
            #         'poisoning/misclassification_rate': 0,
            #     }, step=round)
            # if self.cfg.dataset.clustering:
            #     # todo add clustering metrics
            #     wandb.log({
            #         'clustering/todo': 0,
            #     }, step=round)

            train_duration = time.time() - begin
            progress.add_train_duration(train_duration)
            wandb.log({'durations/train': train_duration}, step=round)

            test_duration = 0
            if self.cfg.run.eval_every != -1 and round % self.cfg.run.eval_every == 0 and round != self.cfg.run.num_rounds - 1:
                wandb.log({'test/total_published_transactions': total_client_contributions}, step=round)
                begin = time.time()
                test_metrics = self.test(round)
                average_test_accuracy = self.print_test_results(test_metrics, round)
                if average_test_accuracy >= self.cfg.run.target_accuracy:
                    logging.info('Re-running test on all clients to verify early stopping')
                    all_clients_test_metrics = self.test(round, use_all_clients=True)
                    all_clients_avg_test_acc = np.average([r['accuracy'] for r in all_clients_test_metrics])
                    if all_clients_avg_test_acc >= self.cfg.run.target_accuracy:
                        logging.info(
                            f'Stopping due to reaching of target accuracy ({all_clients_avg_test_acc} >= {self.cfg.run.target_accuracy})')
                        final_test_metrics = all_clients_test_metrics
                        wandb.log({'train/rounds_to_target_accuracy': round}, step=round)
                        break
                    else:
                        logging.info(
                            f'Continuing due to not reaching target accuracy ({all_clients_avg_test_acc} < {self.cfg.run.target_accuracy})')
                test_duration = time.time() - begin
                progress.add_eval_duration(test_duration)
                logging.info(f'Test duration: {test_duration:.2f}s')
                wandb.log({'durations/test': test_duration}, step=round)

            hours_left, mins_left = progress.eta(round)
            logging.info(
                f'This round took: {train_duration + test_duration:.2f}s - {str(hours_left) + "h " if hours_left > 0 else ""}{mins_left}m left')
            sys.stdout.flush()

        # Final evaluation
        if final_test_metrics is None:
            final_test_metrics = self.test(round, use_all_clients=True)
        self.print_test_results(final_test_metrics, round)
        wandb.log({'train/rounds_to_target_accuracy': round}, step=round)
        return round

    def train_one_round(self, round):
        clients = self.dataset.select_clients(round, self.cfg.run.clients_per_round, self.cfg.run.sample_clients,
                                              self.active_clients_list)
        logging.debug(f"Clients this round: {clients}")

        model_updates = [self.train_one_client(round, client_id) for client_id in clients]

        return model_updates

    def train_one_client(self, round, client_id):
        client_data = self.dataset.get_all_dataset_partitions_for_client(client_id)
        client_model = Lab.create_client_model(self.cfg.seed, self.cfg.run, self.cfg.dataset)
        client_cluster_id = self.dataset.get_cluster_id_for_client(client_id)

        # Choose which nodes are malicious based on a hash, not based on a random variable
        # to have it consistent over the entire experiment run
        # https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
        use_poisoning_node = \
            self.cfg.poisoning.type != "disabled" and \
            self.cfg.poisoning.from_round <= round and \
            (float(crc32(client_id.encode('utf-8')) & 0xffffffff) / 2 ** 32) < self.cfg.poisoning.fraction

        if use_poisoning_node:
            logging.info(
                f'client {client_id} is is poisoned {"and uses random ts" if self.cfg.poisoning.use_random_ts else ""}')
            node = MaliciousNode(None, None, DummyTipSelector(), client_id, client_cluster_id,
                                 client_data, client_model, PoisonType.make_type_from_cfg(self.cfg.poisoning.type),
                                 config=self.cfg.node)
        else:
            node = Node(None, None, DummyTipSelector(), client_id, client_cluster_id, client_data,
                        client_model, None)

        old_params = np.array(self.global_params)
        old_model_metrics = node.test(old_params, 'val')

        new_params = np.array(node.train(self.global_params))

        new_model_metrics = node.test(new_params, 'val')

        metrics = {
            'client_id': client_id,
            'cluster_id': client_cluster_id,
            'val_loss': new_model_metrics['loss'],
            'val_accuracy': new_model_metrics['accuracy'],
            'old_val_loss': old_model_metrics['loss'],
            'old_val_accuracy': old_model_metrics['accuracy']
        }

        return old_params - new_params, self.dataset.get_dataset_size_for_client(client_id), metrics

    def test(self, round, use_all_clients=False):
        logging.info('Test for round %s' % round)

        # select clients - potentially fairly from clusters
        if use_all_clients:
            clients = self.dataset.client_ids
        elif self.dataset.get_cluster_id_for_client(self.dataset.client_ids[0]) == -1:
            # No clusters used
            clients = self.dataset.select_clients(round, self.cfg.run.test_on_fraction, sample_clients=False,
                                                  log_number_of_clients=False)
        else:
            # clusters used
            client_indices = []
            clusters = np.array(list(map(self.dataset.get_cluster_id_for_client, self.dataset.client_ids)))
            unique_clusters = set(clusters)
            num = max(min(int(len(self.dataset.client_ids) * self.cfg.run.test_on_fraction), len(self.dataset.client_ids)), 1)
            div = len(unique_clusters)
            clients_per_cluster = [num // div + (1 if x < num % div else 0) for x in range(div)]
            for cluster_id in unique_clusters:
                cluster_client_ids = np.where(clusters == cluster_id)[0]
                client_indices.extend(
                    np.random.choice(cluster_client_ids, clients_per_cluster[cluster_id], replace=False))
            clients = [self.dataset.client_ids[i] for i in client_indices]
        logging.debug(f"Clients for testing: {clients}")
        return [self.test_single(client_id, random.randint(0, 4294967295)) for client_id in clients]

    def test_single(self, client_id, seed):
        random.seed(1 + seed)
        np.random.seed(12 + seed)
        tf.compat.v1.set_random_seed(123 + seed)

        client_model = Lab.create_client_model(seed, self.cfg.run, self.cfg.dataset)
        client_data = self.dataset.get_all_dataset_partitions_for_client(client_id)
        node = Node(None, None, DummyTipSelector(), client_id, self.dataset.get_cluster_id_for_client(client_id),
                    client_data, client_model)

        metrics = node.test(self.global_params, 'test')

        return metrics


    def print_test_results(self, results, rnd):
        avg_acc = np.average([r['accuracy'] for r in results])
        avg_loss = np.average([r['loss'] for r in results])

        wandb.log({
            'test/accuracy': avg_acc,
            'test/loss': avg_loss
        }, step=rnd)

        logging.info(f'Average accuracy: {avg_acc}\nAverage loss: {avg_loss}')

        if self.cfg.poisoning.type != "disabled":
            # Todo poisoning test logging
            avg_approved_poisoned_transactions = np.average([r['num_approved_poisoned_transactions'] for r in results])
            wandb.log({
                'test/num_approved_poisoned_transactions': avg_approved_poisoned_transactions
            }, step=rnd)
            logging.info(f'Average number of approved poisoned transactions: {avg_approved_poisoned_transactions}')

        import csv
        import os

        write_header = False
        if not os.path.exists(os.path.join(self.cfg.experiment_folder, 'acc_and_loss_all.csv')):
            write_header = True

        with open(os.path.join(self.cfg.experiment_folder, 'acc_and_loss.csv'), 'a', newline='') as f:
            csvwriter = csv.writer(f)
            if write_header:
                csvwriter.writerow(['round', 'accuracy', 'loss'])
            csvwriter.writerow([rnd, avg_acc, avg_loss])

        with open(os.path.join(self.cfg.experiment_folder, 'acc_and_loss_all.csv'), 'a', newline='') as f:
            for r in results:
                r['round'] = rnd
                r['conf_matrix'] = r['conf_matrix'].tolist()

                w = csv.DictWriter(f, r.keys())
                if write_header:
                    w.writeheader()
                    write_header = False

                w.writerow(r)
        return avg_acc

############## Testing ##############

def test_acc_per_cluster(global_params, dataset, model_config, clients_to_test_on, seed, cids):
    accuracies_per_cluster = {}

    for cid in cids:
        accuracies_per_cluster[cid] = []
    
    accuracies = _test_acc_clients(global_params, dataset, model_config, clients_to_test_on, seed)

    for idx, (_, cid) in enumerate(clients_to_test_on):
        accuracies_per_cluster[cid].append(accuracies[idx])
    
    return accuracies_per_cluster

############## Helpers ##############

def get_unique_cluster_ids(clients):
    return list({ cid for (_, cid) in clients })

def plot_accuracy_boxplot(data, cids, print_avg_acc=False):
    # print for each cluster
    for cid in cids:
        cluster_data = [epoch[cid] for epoch in data if cid in epoch]
        _plot_accuracy_boxplot(cluster_data, cid, print_avg_acc)

    # print for all clusters
    all_cluster_data = [sum(epoch.values(), []) for epoch in data]
    print(all_cluster_data)
    with open('fed_avg_accuracy_per_round_all.txt', 'w') as f:
        for round_number, round_data in enumerate(all_cluster_data):
            f.write(f'{round_number+1} {" ".join(map(str,round_data))}\n')

    _plot_accuracy_boxplot(all_cluster_data, "all", print_avg_acc)

def _plot_accuracy_boxplot(data, cid, print_avg_acc, max_y=1):
    last_generation = len(data)

    plt.boxplot(data)

    # Fix y axis data range to [0, 1]
    plt.ylim([0, max_y])

    if print_avg_acc:
        plt.plot([i for i in range(last_generation)], [np.mean(x) for x in data])
    
    # Settings for plot
    plt.title("Accuracy per round (cluster: %s)" % cid)
    
    plt.xlabel("")
    plt.xticks([i for i in range(last_generation)], [i if i % 10 == 0 else '' for i in range(last_generation)])
    
    plt.ylabel("")
    
    analysis_filepath = ("fed_avg_accuracy_per_round_cluster_%s" % cid)
    plt.savefig(analysis_filepath+".png")

    plt.title("")
    plt.savefig(analysis_filepath+".pdf")
    
    plt.clf()