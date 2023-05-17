##### Imports

import json
import os.path

import wandb

from .graph import Graph

class TangleAnalyser:
    def __init__(self, src_tangle_dir, generation, analysis_output_dir):
        self.acc_loss_all_csvfile = os.path.join(os.path.dirname(src_tangle_dir), 'acc_and_loss_all.csv')
        with open(f'{src_tangle_dir}/tangle_{generation}.json', "r") as tf:
            data = json.load(tf)

        self.graph = Graph(data, generation, analysis_output_dir)

    def save_statistics(self, include_reference_statistics=True,
                        include_cluster_statistics=False, include_poisoning_statistics=False):
        self.graph.print_statistics(include_reference_statistics, include_cluster_statistics, include_poisoning_statistics)
        self.graph.plot_transactions_per_round(plot_for_paper=True)
        self.graph.plot_parents_per_round(plot_first_round=False, plot_for_paper=True)
        self.graph.plot_accuracy_boxplot(plot_for_paper=True)
        self.graph.plot_information_gain_approvals(plot_for_paper=True)
        self.graph.plot_modularity_per_round(plot_for_paper=True)
        self.graph.plot_num_modules_per_round(plot_for_paper=True)
        self.graph.plot_misclassification_per_round(plot_for_paper=True)
        self.graph.plot_total_participating_clients_per_round(plot_for_paper=True)
        if include_reference_statistics:
            self.graph.plot_information_gain_ref_tx(plot_for_paper=True)
            self.graph.plot_avg_age_difference_ref_tx(plot_for_paper=True)
            self.graph.plot_pureness_ref_tx(plot_for_paper=True)
            self.graph.plot_pureness_approvals(plot_for_paper=True)
        if include_poisoning_statistics:
            self.graph.plot_poisoning_avg_num_approved_poisoned_tx_in_consensus_per_round(self.acc_loss_all_csvfile, plot_for_paper=True)
            self.graph.plot_poisoning_misclassification_by_confusion_matrix_per_round(self.acc_loss_all_csvfile, plot_for_paper=True)
