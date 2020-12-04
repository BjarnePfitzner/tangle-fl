import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import make_interp_spline, BSpline

from .node import Node

class Graph:
    def __init__(self, data, generation, analysis_output_dir=None):
        self.nodes = [Node(n["id"], n["parents"], n["metadata"]) for n in data["nodes"] if n["metadata"]["time"] <= generation]
        self.generation = generation
        self.analysis_output_dir = analysis_output_dir
    
    def print_statistics(self):
        #### Helper methods for printing
        def _print(text):
            if self.analysis_output_dir:
                analysis_filepath = os.path.join(self.analysis_output_dir, "statistics.txt")
                with open(analysis_filepath, "a+") as statistics_file:
                    print(text, file=statistics_file)
            else:
                print(text)
        
        def _print_multiple_statistics_lines(labels, data, text):
            for l, d in zip(labels, data):
                _print(text % (l, d))
            _print("")

        #### Get statistics data and print it
        statistics = self._get_statistics_data()

        _print("Average clients per round: %f" % statistics["average_clients_per_round"])
        _print("")
        _print("Average parents per round (not including round 1): %f" % statistics["average_parents_per_round"])
        _print("")
        
        # Pureness
        _print_multiple_statistics_lines(
            *statistics["average_pureness_per_round_approvals"],
            "Average pureness (approvals) for %s per round: %f")
        _print_multiple_statistics_lines(
            *statistics["average_pureness_per_round_ref_tx"],
            "Average pureness (ref_tx) for %s per round: %f")
        
        # Information gain
        _print_multiple_statistics_lines(
            *statistics["information_gain_per_round_approvals"],
            "Average information gain (approvals) for %s per round: %f")
        _print_multiple_statistics_lines(
            *statistics["information_gain_per_round_ref_tx"],
            "Average information gain (ref_tx) for %s per round: %f")
        
    def plot_transactions_per_round(self, smooth_line=False):
        data = self._get_num_transactions_per_round()
        
        self._line_plot(
            title='Number of transactions per round',
            data_arrays=[data],
            y_label="Number of transactions",
            smooth_line=smooth_line)

    def plot_parents_per_round(self, plot_first_round=False, smooth_line=False):
        data = self._get_mean_parents_per_round(plot_first_round)
        
        self._line_plot(
            title='Mean number of parents per round (%s round 1)' % ("including" if plot_first_round else "excluding"),
            data_arrays=[data],
            y_label="Mean number of parents",
            smooth_line=smooth_line)
    
    def plot_accuracy_boxplot(self, print_avg_acc=False):
        data = self._prepare_acc_data()
        plt.boxplot(data)

        if print_avg_acc:
            plt.plot([i for i in range(1, self.generation + 1)], [np.mean(x) for x in data])
        
        # Settings for plot
        plt.title('Accuracy per round')
        
        plt.xlabel("Round")        
        plt.xticks([i for i in range(1, self.generation + 1)], [i if i % 5 == 0 else '' for i in range(1, self.generation + 1)])
        
        plt.ylabel("")
        
        if self.analysis_output_dir:
            analysis_filepath = os.path.join(self.analysis_output_dir, "accuracy_per_round.png")
            plt.savefig(analysis_filepath)
        else:
            plt.show()
        
        plt.clf()
        
    def plot_information_gain_ref_tx(self, smooth_line=False):      
        labels, data_arrays = self._get_information_gain_ref_tx()
        
        # Plot data
        self._line_plot(
            title='Information gain (reference tx)',
            data_arrays=data_arrays,
            labels=labels,
            smooth_line=smooth_line)
        
    def plot_information_gain_app(self, smooth_line=False):
        labels, data_arrays = self._get_information_gain_approvals()
        
        # Plot data
        self._line_plot(
            title='Information gain (approvals)',
            data_arrays=data_arrays,
            labels=labels,
            smooth_line=smooth_line)
    
    def plot_reference_pureness_ref_tx(self, smooth_line=False):
        labels, data_arrays = self._prepare_reference_pureness(compare_to_ref_tx=True)
        
        self._line_plot(
            title='Cluster pureness (reference transaction)',
            data_arrays=data_arrays,
            labels=labels,
            smooth_line=smooth_line)
        
    def plot_reference_pureness_approvals(self, smooth_line=False):
        labels, data_arrays = self._prepare_reference_pureness()
        
        self._line_plot(
            title='Cluster pureness (approvals)',
            data_arrays=data_arrays,
            labels=labels,
            smooth_line=smooth_line)
        
    def plot_avg_age_difference_ref_tx(self, smooth_line=False):
        avg_age_difference_per_round = self._prepare_data_avg_age_difference_to_ref_tx()
        
        self._line_plot(
            title='Average age difference to reference transaction per round',
            data_arrays=[avg_age_difference_per_round],
            y_label='Age in rounds',
            smooth_line=smooth_line)
    
    ##### Private: plots
    
    def _line_plot(self,
                   title,
                   data_arrays,
                   x_label="Round",
                   y_label="",
                   labels=(),
                   smooth_line=False):      
        plt.title(title)
        
        end_index = self.generation + 1
        # All arrays in data_arrays are expected to be of the same length
        # Use a start_index, as some data lines do not start from round 1 (e.g. mean number of parents)
        start_index = end_index - len(data_arrays[0])
        
        x_data_orig = np.array([i for i in range(start_index, end_index)])
        
        for data in data_arrays:
            # The x positions (x_data) need to be of the same length as data
            # As there may be data shorter than self.generation adapt the size of x_data
            x_data = x_data_orig[:len(data)]
            
            if smooth_line:
                # in case of smooth_line define x as 200 equally spaced values between all generations
                x_spaced = np.linspace(x_data.min(), x_data.max(), 200)
                spl = make_interp_spline(x_data, data, k=7)
                y_smooth = spl(x_spaced)
                plt.plot(x_spaced, y_smooth)
            else:
                plt.plot(x_data, data)
        
        if (len(labels) > 0):
            plt.legend(labels)
        
        plt.xlabel(x_label)
        plt.xticks([i for i in range(start_index, end_index)], [i if i % 5 == 0 else '' for i in range(start_index, end_index)])
        
        plt.ylabel(y_label)
        
        if self.analysis_output_dir:
            diagram_filename = title.replace(" ", "_").lower() + ".png"
            diagram_filepath = os.path.join(self.analysis_output_dir, diagram_filename)
            plt.savefig(diagram_filepath)
        else:
            plt.show()
    
        plt.clf()
    
    #### Private: Data preparation

    def _get_statistics_data(self):
        statistics = {}
        
        # Clients and parents
        statistics["average_clients_per_round"] = (len(self.nodes) - 1) / self.generation
        # Divide by len(self.nodes) - 1, because we don't include genesis transaction
        statistics["average_parents_per_round"] = sum([len(n.parents) for n in self.nodes]) / (len(self.nodes) - 1)
        
        # Reference pureness
        labels, data = self._prepare_reference_pureness()
        data = [np.nanmean(pureness) for pureness in data]
        statistics["average_pureness_per_round_approvals"] = (labels, data)
        
        labels, data = self._prepare_reference_pureness(compare_to_ref_tx=True)
        data = [np.nanmean(pureness) for pureness in data]
        statistics["average_pureness_per_round_ref_tx"] = (labels, data)
        
        # Information gain
        labels, data = self._get_information_gain_ref_tx()
        data = [np.nanmean(info_gain) for info_gain in data]
        statistics["information_gain_per_round_ref_tx"] = (labels, data)
        
        labels, data = self._get_information_gain_approvals()
        data = [np.nanmean(info_gain) for info_gain in data]
        statistics["information_gain_per_round_approvals"] = (labels, data)
        
        return statistics
    
    def _get_information_gain_ref_tx(self):
        labels = (
            'Avg "accuracy"',
            'Avg "reference_tx_accuracy"',
            'Avg information_gain')
        
        # Data per node
        acc_per_node = self._prepare_acc_data()
        ref_acc_per_node = self._prepare_ref_acc_data()
        info_gain_per_node = np.array(acc_per_node) - np.array(ref_acc_per_node)
        
        # Aggregate data per round
        avg_acc_per_round = [np.mean(x) for x in acc_per_node]
        avg_ref_acc_per_round = [np.mean(x) for x in ref_acc_per_node]
        avg_info_gain_per_node = [np.mean(x) for x in info_gain_per_node]
        
        return labels, [avg_acc_per_round, avg_ref_acc_per_round, avg_info_gain_per_node]
    
    def _get_information_gain_approvals(self):
        labels = (
            'Avg "accuracy"',
            'Avg "averaged_accuracy"',
            'Avg information_gain')
        
        # Data per node
        acc_per_node = self._prepare_acc_data()
        app_acc_per_node = self._prepare_app_acc_data()
        info_gain_per_node = np.array(acc_per_node) - np.array(app_acc_per_node)
        
        # Aggregate data per round
        avg_acc_per_round = [np.mean(x) for x in acc_per_node]
        avg_app_acc_per_round = [np.mean(x) for x in app_acc_per_node]
        avg_info_gain_per_node = [np.mean(x) for x in info_gain_per_node]
        
        return labels, [avg_acc_per_round, avg_app_acc_per_round, avg_info_gain_per_node]
    
    def _prepare_app_acc_data(self):
        return self._prepare_data_for("averaged_accuracy")
    
    def _prepare_ref_acc_data(self):
        return self._prepare_data_for("reference_tx_accuracy")
    
    def _prepare_acc_data(self):
        return self._prepare_data_for("accuracy")
    
    def _prepare_data_for(self, key):
        """
        Returns a list of arrays.
        Each array represents one time step.
        Each array contains the associated meatadata for each node for that round.
        """
        plot_data = {}

        for n in self.nodes:
            t = n.metadata["time"]
            # Do not contain genesis block
            if t == 0:
                continue
            
            if t not in plot_data:
                plot_data[t] = np.array([])
            
            plot_data[t] = np.append(plot_data[t], n.metadata[key])
        
        return list(plot_data.values())
    
    def _prepare_data_avg_age_difference_to_ref_tx(self):
        avg_age_difference_per_round = []

        # Skip genesis-round
        for i in range(1, self.generation + 1):
            nodes_in_round = self._get_all_nodes_for_time(i)

            # If no nodes were used this round, skip it and assign np.nan as age
            # np.nan => no data available
            if len(nodes_in_round) == 0:
                avg_age_difference_per_round.append(np.nan)
                continue

            summed_age = 0

            for n in nodes_in_round:
                cid = n.metadata["clusterId"]
                ref_tx = n.metadata["reference_tx"]
                ref_tx = next(n for n in self.nodes if n.id == ref_tx)

                summed_age += (n.metadata["time"] - ref_tx.metadata["time"])

            avg_age_difference_per_round.append(summed_age / len(nodes_in_round))
        
        return avg_age_difference_per_round
    
    def _prepare_reference_pureness(self, compare_to_ref_tx=False):
        cids = self._get_unique_cluster_ids()

        # Dict to store pureness data for each cluster
        cluster_data = {}
        for cid in cids:
            cluster_data[cid] = []
            
        # Assumption: there is no cluster with the id "combined"
        cluster_data["combined"] = []

        for r in range(1, self.generation + 1):
            nodes_in_round = self._get_all_nodes_for_time(r)

            # Count for this round for all clusters how many parents have the same cid as their children
            same_cluster_id_round = 0

            for cid in cids:
                nodes_in_round_with_cid = [n for n in nodes_in_round if n.metadata["clusterId"] == cid]

                # If no nodes of this cluster were used this round, skip cluster and assign np.nan as pureness
                # np.nan => no data available
                if len(nodes_in_round_with_cid) == 0:
                    cluster_data[cid].append(np.nan)
                    continue

                parents_with_same_cid = 0

                # For each node in this round with the current cid:
                # check for each parent, if it has the same cluster id
                for n in nodes_in_round_with_cid:
                    ref_transactions = n.parents
                    
                    if compare_to_ref_tx:
                        ref_transactions = [n.metadata["reference_tx"]]
    
                    for p in ref_transactions:
                        p_tx = next(n for n in self.nodes if n.id == p)

                        # In case p is the genesis transaction assign it as the same cluster
                        # (genesis transaction has no cluster id)
                        if "clusterId" in p_tx.metadata:
                            p_tx_cid = p_tx.metadata["clusterId"]
                        else:
                            p_tx_cid = cid

                        if p_tx_cid == cid:
                            parents_with_same_cid += 1

                # Add the number of nodes with the same cluster id as their reference transaction to the global count,
                # so that we can later calculate the combined pureness
                same_cluster_id_round += parents_with_same_cid

                # Add the cluster pureness for this round
                num_refs = sum([len(n.parents) for n in nodes_in_round_with_cid])
                if compare_to_ref_tx:
                    num_refs = len(nodes_in_round_with_cid)
                cluster_data[cid].append(parents_with_same_cid / num_refs)

            # Calculate combined cluster data for this round
            num_refs = sum([len(n.parents) for n in nodes_in_round])
            if compare_to_ref_tx:
                num_refs = len(nodes_in_round)
            cluster_data["combined"].append(same_cluster_id_round / num_refs)

        labels = []
        data_arrays = []

        for label, data in cluster_data.items():
            labels.append("Cluster %s" % label)
            data_arrays.append(data)

        return labels, data_arrays
    
    #### Private: Helpers
    
    def _get_unique_cluster_ids(self):
        cids = set()

        for n in self.nodes:
            if "clusterId" in n.metadata:
                cids.add(n.metadata["clusterId"])
        
        return list(cids)
    
    def _get_all_nodes_for_time(self, time):
        return [n for n in self.nodes if n.metadata["time"] == time]
    
    def _get_num_transactions_per_round(self):
        transactions_per_round = []
        
        # Skip genesis-round
        for r in range(1, self.generation + 1):
            transactions_per_round.append(len(self._get_all_nodes_for_time(r)))
        
        return transactions_per_round

    def _get_mean_parents_per_round(self, plot_first_round):
        mean_parents_per_round = []

        # Skip genesis-round (round 0)
        # Nodes in round 1 will always have parents = ["id-of-genesis-node"]
        if plot_first_round:
            start_gen = 1
        else:
            start_gen = 2
        
        for r in range(start_gen, self.generation + 1):
            nodes = self._get_all_nodes_for_time(r)
            mean_parents_per_round.append(np.mean([len(n.parents) for n in nodes]))
        
        return mean_parents_per_round
