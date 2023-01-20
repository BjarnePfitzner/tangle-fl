##### Imports

import json
import wandb

from .graph import Graph

class TangleAnalysator:
    def __init__(self, src_tangle_dir, generation, analysis_output_dir, wandb_id):
        with open(f'{src_tangle_dir}/tangle_{generation}.json', "r") as tf:
            data = json.load(tf)

        self.graph = Graph(data, generation, analysis_output_dir)

        wandb.init(project="Tangle", entity="bjarnepfitzner", id=wandb_id, resume='must')

    def save_statistics(self, include_reference_statistics=True):
        self.graph.print_statistics(include_reference_statistics)
        self.graph.plot_transactions_per_round(plot_for_paper=True)
        self.graph.plot_parents_per_round(plot_first_round=False, plot_for_paper=True)
        self.graph.plot_accuracy_boxplot(plot_for_paper=True)
        self.graph.plot_information_gain_approvals(plot_for_paper=True)
        self.graph.plot_modularity_per_round(plot_for_paper=True)
        self.graph.plot_num_modules_per_round(plot_for_paper=True)
        # self.graph.plot_misclassification_per_round(plot_for_paper=True)
        self.graph.plot_total_participating_clients_per_round(plot_for_paper=True)
        if include_reference_statistics:
            self.graph.plot_information_gain_ref_tx(plot_for_paper=True)
            self.graph.plot_avg_age_difference_ref_tx(plot_for_paper=True)
            self.graph.plot_pureness_ref_tx(plot_for_paper=True)
            self.graph.plot_pureness_approvals(plot_for_paper=True)
        wandb.log({})
