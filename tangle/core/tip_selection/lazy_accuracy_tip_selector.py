from tangle.core.tip_selection.tip_selector import TipSelector
from .accuracy_tip_selector import AccuracyTipSelector, AccuracyTipSelectorSettings

from ...models.baseline_constants import ACCURACY_KEY

class LazyAccuracyTipSelector(AccuracyTipSelector):
    def __init__(self, tangle, tip_selection_settings, particle_settings):
        super().__init__(tangle, tip_selection_settings, particle_settings)
        self.ratings = {}

    def tx_rating(self, tx, node):

        if node.id in self.ratings and tx in self.ratings[node.id]:
            return self.ratings[node.id][tx]

        if not(node.id in self.ratings and tx in self.ratings[node.id]):
            self.compute_ratings_lazy(node, tx)
        
        return self.ratings[node.id][tx]

    ########################################

    def compute_ratings(self, node):
        # Do not calculate ratings yet, because we are lazy
        # But initialize the cache
        if node.id not in self.ratings:
            self.ratings[node.id] = {}

    def compute_ratings_lazy(self, node, tx):
        rating = self._compute_ratings(node, tx)

        if self.settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS]:
            def cumulate_ratings(future_set, ratings):
                cumulated = 0
                for tx_id in future_set:
                    cumulated += ratings[tx_id]
                return cumulated

            future_set_cache = {}
            for tx_id in self.tangle.transactions:
                future_set = TipSelector.future_set(tx_id, self.approving_transactions, future_set_cache)
                rating[tx_id] = cumulate_ratings(future_set, rating) + rating[tx_id]

        self.ratings[node.id].update(rating)

    def _compute_ratings(self, node, tx):
        rating = {}

        if self.settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] == "GLOBAL":
            txs = self.tips
            if tx not in txs:
                txs.append(tx)
        elif self.settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS]:
            future_set_cache = {}
            txs = TipSelector.future_set(tx, self.approving_transactions, future_set_cache)
        else:
            txs = [tx]

        for tx_id in txs:
            rating[tx_id] = node.test(node.tx_store.load_transaction_weights(tx_id), 'train', True)[ACCURACY_KEY]

        # We (currently) do not care about the future-set-size-based rating
        # future_set_cache = {}
        # for tx in txs:
        #     rating[tx] *= len(TipSelector.future_set(tx, self.approving_transactions, future_set_cache)) + 1

        return rating
