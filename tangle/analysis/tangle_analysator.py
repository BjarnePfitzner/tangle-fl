##### Imports

import json

from .graph import Graph

class TangleAnalysator:
    def __init__(self, src_tangle_dir, generation, analysis_output_dir):
        with open(f'{src_tangle_dir}/tangle_{generation}.json', "r") as tf:
            data = json.load(tf)
        
        self.graph = Graph(data, generation, analysis_output_dir)
    
    def save_statistics(self):
        self.graph.print_statistics()
        self.graph.plot_transactions_per_round()
        self.graph.plot_parents_per_round(plot_first_round=False)
        self.graph.plot_accuracy_boxplot()
        self.graph.plot_information_gain_ref_tx()
        self.graph.plot_information_gain_app()
        self.graph.plot_reference_pureness_ref_tx()
        self.graph.plot_reference_pureness_approvals()
        self.graph.plot_avg_age_difference_ref_tx()
