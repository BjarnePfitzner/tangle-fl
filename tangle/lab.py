import random
import os
import time
import sys
import itertools

import wandb
import numpy as np
import importlib
import logging
from zlib import crc32

from data.abstract_dataset import AbstractDataset
from differential_privacy import RDPAccountant
from tangle.core import Transaction, Node, MaliciousNode, PoisonType
from tangle.core.tip_selection import TipSelector
from tangle.tip_selector_factory import TipSelectorFactory
from tangle.lab_transaction_store import LabTransactionStore
from tangle.core.tangle import Tangle
from tangle.progress import Progress


class Lab:
    def __init__(self, tip_selector_factory: TipSelectorFactory, dataset: AbstractDataset, cfg, tx_store=None):
        self.tip_selector_factory = tip_selector_factory
        self.dataset = dataset
        self.active_clients_list = self.dataset.client_ids[:]
        self.cfg = cfg
        self.tx_store = tx_store if tx_store is not None else LabTransactionStore(os.path.join(self.cfg.experiment_folder, self.cfg.tangle_dir), self.cfg.load_tangle_from)

        # Set the random seed if provided (affects client sampling, and batching)
        random.seed(1 + cfg.seed)
        np.random.seed(12 + cfg.seed)

        # Setup tangle
        if self.cfg.run.start_from_round > 0:
            tangle_name = int(self.cfg.run.start_from_round) - 1
            logging.info('Loading previous tangle from round %s' % tangle_name)
            self.tangle = self.tx_store.load_tangle(tangle_name)
        else:
            genesis = self.create_genesis()
            self.tangle = Tangle({genesis.id: genesis}, genesis.id)

            test_metrics = self.test(0)
            self.print_test_results(test_metrics, 0)

            self.cfg.run.start_from_round = 1

    def run_training(self):
        if self.cfg.run.num_rounds == -1:
            rounds_iter = itertools.count(self.cfg.run.start_from_round)
            progress = Progress(1000, self.cfg.run.eval_every)
        else:
            rounds_iter = range(self.cfg.run.start_from_round, self.cfg.run.num_rounds)
            progress = Progress(self.cfg.run.num_rounds - self.cfg.run.start_from_round, self.cfg.run.eval_every)

        # todo if continuing this does not work
        all_publishing_clients = set([])
        total_published_transactions = 0
        final_test_metrics = None

        for round in rounds_iter:
            begin = time.time()
            logging.info('Started training for round %s' % round)
            sys.stdout.flush()

            new_txs = self.train_one_round(round)
            for tx in new_txs:
                if tx is not None:
                    self.tangle.add_transaction(tx)
                    total_published_transactions += 1
            all_publishing_clients = all_publishing_clients.union(set([tx.metadata['client_id'] for tx in new_txs if tx is not None]))

            # Log some metrics
            wandb.log({
                'train/fraction_publishing_tx': len([tx for tx in new_txs if tx is not None]) / len(new_txs),
                'train/avg_num_parents': np.mean([len(tx.parents) for tx in new_txs if tx is not None]),
                'train/total_number_publishing_clients': len(all_publishing_clients),
                'val/loss': wandb.Histogram([tx.metadata['loss'] for tx in new_txs if tx is not None]),
                'val/loss_avg': np.mean([tx.metadata['loss'] for tx in new_txs if tx is not None]),
                'val/accuracy': wandb.Histogram([tx.metadata['accuracy'] for tx in new_txs if tx is not None]),
                'val/accuracy_avg': np.mean([tx.metadata['accuracy'] for tx in new_txs if tx is not None]),
                'val/averaged_parents_loss': wandb.Histogram([tx.metadata['averaged_loss'] for tx in new_txs if tx is not None]),
                'val/averaged_parents_loss_avg': np.mean([tx.metadata['averaged_loss'] for tx in new_txs if tx is not None]),
                'val/averaged_parents_accuracy': wandb.Histogram([tx.metadata['averaged_accuracy'] for tx in new_txs if tx is not None]),
                'val/averaged_parents_accuracy_avg': np.mean([tx.metadata['averaged_accuracy'] for tx in new_txs if tx is not None]),
                'information_gain/parent_txs': wandb.Histogram([tx.metadata['accuracy'] - tx.metadata['averaged_accuracy'] for tx in new_txs if tx is not None]),
                'information_gain/parent_txs_avg': np.mean([tx.metadata['accuracy'] - tx.metadata['averaged_accuracy'] for tx in new_txs if tx is not None]),
            }, step=round)
            if self.cfg.node.publish_if_better_than == 'REFERENCE':
                wandb.log({
                    'train/avg_age_diff_to_ref_tx': np.mean([round - self.tangle.transactions[tx.metadata['reference_tx']].metadata['time'] for tx in new_txs if tx is not None]),
                    'val/reference_tx_loss': wandb.Histogram([tx.metadata['reference_tx_loss'] for tx in new_txs if tx is not None]),
                    'val/reference_tx_loss_avg': np.mean([tx.metadata['reference_tx_loss'] for tx in new_txs if tx is not None]),
                    'val/reference_tx_accuracy': wandb.Histogram([tx.metadata['reference_tx_accuracy'] for tx in new_txs if tx is not None]),
                    'val/reference_tx_accuracy_avg': np.mean([tx.metadata['reference_tx_accuracy'] for tx in new_txs if tx is not None]),
                    'information_gain/ref_tx_acc': wandb.Histogram([tx.metadata['accuracy'] - tx.metadata['reference_tx_accuracy'] for tx in new_txs if tx is not None]),
                    'information_gain/ref_tx_acc_avg': np.mean([tx.metadata['accuracy'] - tx.metadata['reference_tx_accuracy'] for tx in new_txs if tx is not None]),
                    'information_gain/ref_tx_loss': wandb.Histogram([tx.metadata['reference_tx_loss'] - tx.metadata['loss'] for tx in new_txs if tx is not None]),
                    'information_gain/ref_tx_loss_avg': np.mean([tx.metadata['reference_tx_loss'] - tx.metadata['loss'] for tx in new_txs if tx is not None]),
                }, step=round)
            if self.cfg.poisoning.type != 'disabled':
                # todo add poisoning metrics
                wandb.log({
                    'poisoning/misclassification_rate': 0,
                }, step=round)
            if self.cfg.dataset.clustering:
                # todo add clustering metrics
                wandb.log({
                    'clustering/todo': 0,
                }, step=round)

            train_duration = time.time() - begin
            progress.add_train_duration(train_duration)
            wandb.log({'durations/train': train_duration}, step=round)

            self.tx_store.save_tangle(self.tangle, round)

            test_duration = 0
            if self.cfg.run.eval_every != -1 and round % self.cfg.run.eval_every == 0 and round != self.cfg.run.num_rounds - 1:
                wandb.log({'test/total_published_transactions': total_published_transactions}, step=round)
                begin = time.time()
                test_metrics = self.test(round)
                average_test_accuracy = self.print_test_results(test_metrics, round)
                if average_test_accuracy >= self.cfg.run.target_accuracy:
                    logging.info('Re-running test on all clients to verify early stopping')
                    all_clients_test_metrics = self.test(round, use_all_clients=True)
                    average_test_accuracy = np.average([r['accuracy'] for r in all_clients_test_metrics])
                    if average_test_accuracy >= self.cfg.run.target_accuracy:
                        logging.info(f'Stopping due to reaching of target accuracy ({average_test_accuracy} >= {self.cfg.run.target_accuracy})')
                        final_test_metrics = all_clients_test_metrics
                        wandb.log({'train/rounds_to_target_accuracy': round}, step=round)
                        break
                    else:
                        logging.info(f'Continuing due to not reaching target accuracy ({average_test_accuracy} < {self.cfg.run.target_accuracy})')
                test_duration = time.time() - begin
                progress.add_eval_duration(test_duration)
                logging.info(f'Test duration: {test_duration:.2f}s')
                wandb.log({'durations/test': test_duration}, step=round)

            hours_left, mins_left = progress.eta(round)
            logging.info(f'This round took: {train_duration + test_duration:.2f}s - {str(hours_left) + "h " if hours_left > 0 else ""}{mins_left}m left')
            sys.stdout.flush()

        # Final evaluation
        if final_test_metrics is None:
            final_test_metrics = self.test(round, use_all_clients=True)
        self.print_test_results(final_test_metrics, round)
        wandb.log({'train/rounds_to_target_accuracy': round}, step=round)
        return round

    @staticmethod
    def create_client_model(seed, run_config, dataset_config):
        model_path = '.%s.%s' % (dataset_config.name, run_config.model)
        mod = importlib.import_module(model_path, package='models')
        ClientModel = getattr(mod, 'ClientModel')

        model = ClientModel(seed, run_config.lr, dataset_config)
        return model

    def create_genesis(self):
        import tensorflow as tf
        logging.debug('Creating genesis transaction')
        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        logging.debug('Creating client model')
        client_model = self.create_client_model(self.cfg.seed, self.cfg.run, self.cfg.dataset)

        genesis = Transaction([])
        genesis.add_metadata('time', 0)
        logging.debug('saving tangle')
        self.tx_store.save(genesis, client_model.get_params())

        return genesis

    def train_one_round(self, round):
        clients = self.dataset.select_clients(round, self.cfg.run.clients_per_round, self.cfg.run.sample_clients,
                                              self.active_clients_list)
        logging.debug(f"Clients this round: {clients}")

        tip_selectors = [self.tip_selector_factory.create(self.tangle) for _ in range(len(clients))]

        result = [self.train_one_client(round, client_id, tip_selector)
                  for (client_id, tip_selector) in zip(clients, tip_selectors)]

        for tx, tx_weights in result:
            if tx is not None:
                self.tx_store.save(tx, tx_weights)

        return [tx for tx, _ in result]

    def train_one_client(self, round, client_id, tip_selector):
        client_data = self.dataset.get_all_dataset_partitions_for_client(client_id)
        client_model = Lab.create_client_model(self.cfg.seed, self.cfg.run, self.cfg.dataset)
        client_cluster_id = self.dataset.get_cluster_id_for_client(client_id)

        # Choose which nodes are malicious based on a hash, not based on a random variable
        # to have it consistent over the entire experiment run
        # https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
        use_poisoning_node = \
            self.cfg.poisoning.type != "disabled" and \
            self.cfg.poisoning.from_round <= round and \
            (float(crc32(client_id.encode('utf-8')) & 0xffffffff) / 2**32) < self.cfg.poisoning.fraction

        if use_poisoning_node:
            ts = TipSelector(self.tangle, particle_settings=self.tip_selector_factory.particle_settings) \
                if self.cfg.poisoning.use_random_ts else tip_selector
            logging.info(f'client {client_id} is is poisoned {"and uses random ts" if self.cfg.poisoning.use_random_ts else ""}')
            node = MaliciousNode(self.tangle, self.tx_store, ts, client_id, client_cluster_id,
                                 client_data, client_model, PoisonType.make_type_from_cfg(self.cfg.poisoning.type), config=self.cfg.node)
        else:
            node = Node(self.tangle, self.tx_store, tip_selector, client_id, client_cluster_id, client_data, client_model, config=self.cfg.node)

        tx, tx_weights = node.create_transaction()

        if tx is not None:
            tx.add_metadata('time', round)

        return tx, tx_weights

    def test(self, round, use_all_clients=False):
        logging.info('Test for round %s' % round)
        tip_selector = self.tip_selector_factory.create(self.tangle)

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
        return [self.test_single(client_id, random.randint(0, 4294967295), tip_selector) for client_id in clients]

    def test_single(self, client_id, seed, tip_selector):
        import tensorflow as tf

        random.seed(1 + seed)
        np.random.seed(12 + seed)
        tf.compat.v1.set_random_seed(123 + seed)

        client_model = self.create_client_model(seed, self.cfg.run, self.cfg.dataset)
        node = Node(self.tangle, self.tx_store, tip_selector,
                    client_id, self.dataset.get_cluster_id_for_client(client_id),
                    self.dataset.get_all_dataset_partitions_for_client(client_id),
                    client_model, config=self.cfg.node)

        reference_txs, reference = node.obtain_reference_params(self.cfg.node.test_reference_avg_top)
        metrics = node.test(reference, 'test')

        if self.cfg.poisoning.type != "disabled":
            # How many unique poisoned transactions have found their way into the consensus
            # through direct or indirect approvals?

            approved_poisoned_transactions_cache = {}

            def compute_approved_poisoned_transactions(transaction):
                if transaction not in approved_poisoned_transactions_cache:
                    tx = self.tangle.transactions[transaction]
                    result = {transaction} if 'poisoned' in tx.metadata and tx.metadata['poisoned'] else set([])
                    result = result.union(*[compute_approved_poisoned_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
                    approved_poisoned_transactions_cache[transaction] = result

                return approved_poisoned_transactions_cache[transaction]

            approved_poisoned_transactions = set(*[compute_approved_poisoned_transactions(tx) for tx in reference_txs])
            metrics['num_approved_poisoned_transactions'] = len(approved_poisoned_transactions)

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


def setup_rdp_accountant(cfg, dataset):
    if cfg.differential_privacy.enabled:
        start_time = time.time()
        rdp_accountants = {
            client_id: RDPAccountant(
                q=min(1.0, cfg.training.batch_size / dataset.get_dataset_size_for_client(client_id)),
                z=cfg.differential_privacy.noise_multiplier,
                N=dataset.get_dataset_size_for_client(client_id),
                dp_type=cfg.differential_privacy.type,
                max_eps=cfg.differential_privacy.epsilon,
                target_delta=cfg.differential_privacy.delta)
            for client_id in dataset.client_ids}
        mid_time = time.time()
        logging.debug(f'Setting up local RDP Accountants took {mid_time - start_time:.3f}s')
        local_steps_cache = {}

        def cached_max_local_steps(rdp_accountant: RDPAccountant):
            if rdp_accountant.q in local_steps_cache.keys():
                return local_steps_cache[rdp_accountant.q]
            max_n_steps, _, _ = rdp_accountant.get_maximum_n_steps(base_n_steps=0,
                                                                   n_steps_increment=10)
            local_steps_cache[rdp_accountant.q] = max_n_steps
            return max_n_steps

        max_local_steps = list(map(cached_max_local_steps, rdp_accountants.values()))
        logging.info(f'Maximum local steps: {np.mean(max_local_steps)} +/- {np.std(max_local_steps)}')
        logging.debug(f'Calculating max local steps took {time.time() - mid_time:.3f}s')
        if np.mean(max_local_steps) < 1.0:
            logging.info(f'Stopping run due to no local privacy budget.')
            wandb.log({'test/accuracy': 0.0,
                       'train/rounds_to_target_accuracy': -1}, step=0)
            exit(0)
        return rdp_accountants, max_local_steps
