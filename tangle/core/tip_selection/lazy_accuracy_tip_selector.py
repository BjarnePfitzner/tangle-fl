from .accuracy_tip_selector import AccuracyTipSelector, AccuracyTipSelectorSettings

from ...models.baseline_constants import ACCURACY_KEY

class LazyAccuracyTipSelector(AccuracyTipSelector):
    def __init__(self, tangle, tip_selection_settings, particle_settings):
        super().__init__(tangle, tip_selection_settings, particle_settings)
        # a cache holding accuracy for transactions
        self.accuracies = {}
        # holds the actual ratings (e.g. in case of CUMULATE_RATINGS cumulated accuracies)
        self.ratings = {}

    def compute_ratings(self, node):
        # Do not calculate ratings yet, because we are lazy
        pass

    def tx_rating(self, tx, node):
        if node.id in self.ratings and tx in self.ratings[node.id]:
            return self.ratings[node.id][tx]
        
        self._calculate_tx_ratings(tx, node)
        return self.ratings[node.id][tx]

    def _calculate_tx_ratings(self, tx, node):
        if node.id not in self.accuracies:
            self.accuracies[node.id] = {}
        if node.id not in self.ratings:
            self.ratings[node.id] = {}

        txs_to_eval = [tx]

        # If we cumulate ratings, get all future transactions
        future_set_cache = {}
        if self.settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS]:
            txs_to_eval.extend(super().future_set(tx, self.approving_transactions, future_set_cache))

        # Calculate accuracies for tx and its future transactions or get them from cache
        accuracies = self._get_or_calculate_accuracies(txs_to_eval, node, future_set_cache)

        # Save calculated accuracies to cache
        for (t_id, acc) in dict(zip(txs_to_eval, accuracies)).items():
            self.accuracies[node.id][t_id] = acc

        # For each transaction in txs_to_eval calculate the rating
        # This is no great computational overhead, because we already calculated all necessary accuracies and future sets
        for t in txs_to_eval:
            rating = self.accuracies[node.id][t]

            if self.settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS]:
                future_txs = super().future_set(t, self.approving_transactions, future_set_cache)
            else:
                future_txs = []
            
            for ft in future_txs:
                rating += self.accuracies[node.id][ft]
            
            self.ratings[node.id][t] = rating
    
    def _get_or_calculate_accuracies(self, txs_to_eval, node, future_set_cache):
        def _get_or_calculate_acc(tx, node,future_set_cache):
            if node.id in self.accuracies and tx in self.accuracies[node.id]:
                return self.accuracies[node.id][tx]

            acc = node.test(node.tx_store.load_transaction_weights(tx), 'train')[ACCURACY_KEY]
            # Multiply size of tree behind this node (see `_compute_ratings` in accuracy_tip_selector.py)
            acc *= len(super().future_set(tx, self.approving_transactions, future_set_cache)) + 1

            return acc

        return [_get_or_calculate_acc(t, node, future_set_cache) for t in txs_to_eval]
