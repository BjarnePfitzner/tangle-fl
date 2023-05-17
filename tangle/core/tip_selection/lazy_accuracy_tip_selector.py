from tangle.core.tip_selection import TipSelector, AccuracyTipSelector
from tangle.core.tip_selection.tip_selection_settings import TipSelectorSettings


class LazyAccuracyTipSelector(AccuracyTipSelector):
    def __init__(self, tangle, tip_selection_settings, particle_settings):
        super().__init__(tangle, tip_selection_settings, particle_settings)
        self.ratings = {}

    def compute_ratings(self, node):
        # Do not calculate ratings yet, because we are lazy
        # But initialize the cache
        if node.client_id not in self.ratings:
            self.ratings[node.client_id] = {}

    def tx_rating(self, tx, node):
        # If the transaction is not in the cache, calculate it's rating first
        if (node.client_id not in self.ratings) or (tx not in self.ratings[node.client_id]):
            super(LazyAccuracyTipSelector, self).compute_ratings(node, tx)
        
        return self.ratings[node.client_id][tx]

    #### Implemented template methods from AccuracyTipSelector
    
    def _update_ratings(self, node_id, rating):
        # Use update (instead of replacing), because rating will not include already computed accuracies
        self.ratings[node_id].update(rating)
    
    def _get_transactions_to_compute(self, tx):
        if self.settings[TipSelectorSettings.CUMULATE_RATINGS]:
            future_set_cache = {}
            future_set = TipSelector.future_set(tx, self.approving_transactions, future_set_cache)
            future_set.add(tx)
            return future_set
        
        return [tx]
