from enum import Enum

from .tip_selector import TipSelector, TipSelectorSettings

from ...models.baseline_constants import ACCURACY_KEY

# Adopted from https://docs.iota.org/docs/node-software/0.1/iri/references/iri-configuration-options

class AccuracyTipSelectorSettings(Enum):
    SELECTION_STRATEGY = 0
    CUMULATE_RATINGS = 1
    RATINGS_TO_WEIGHT = 2
    ALPHA = 3
    SELECT_FROM_WEIGHTS = 4

class AccuracyTipSelector(TipSelector):
    def __init__(self, tangle, tip_selection_settings, particle_settings={TipSelectorSettings.USE_PARTICLES: False}):
        super().__init__(tangle, particle_settings=particle_settings)
        self.settings = tip_selection_settings

        self.tips = []
        for x, tx in self.tangle.transactions.items():
            if len(self.approving_transactions[x]) == 0:
                self.tips.append(x)

    def tip_selection(self, num_tips, node):
        if self.settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] == "GLOBAL":
            self.tips.sort(key=lambda x: self.tx_rating(x, node), reverse=True)
            return self.tips[0:num_tips]
        else:
            return super(AccuracyTipSelector, self).tip_selection(num_tips, node)

    def _compute_ratings(self, node):
        rating = {}

        if self.settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] == "GLOBAL" and not self.settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS]:
            txs = self.tips
        else:
            txs = self.tangle.transactions.keys()

        for tx_id in txs:
            rating[tx_id] = node.test(node.tx_store.load_transaction_weights(tx_id), 'train', True)[ACCURACY_KEY]

        # future_set_cache = {}
        # for tx in txs:
        #     rating[tx] *= len(TipSelector.future_set(tx, self.approving_transactions, future_set_cache)) + 1

        return rating

    def compute_ratings(self, node):
        print(f"computing ratings for node {node.id}")
        rating = self._compute_ratings(node)

        if self.settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS]:
            def cumulate_ratings(future_set, ratings):
                cumulated = 0
                for tx_id in future_set:
                    cumulated += ratings[tx_id]
                return cumulated

            future_set_cache = {}
            for tx_id in self.tangle.transactions:
                future_set = super().future_set(tx_id, self.approving_transactions, future_set_cache)
                rating[tx_id] = cumulate_ratings(future_set, rating) + rating[tx_id]

        print("done computing ratings")
        self.ratings = rating

    def ratings_to_weight(self, ratings, alpha=None):
        if self.settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] == 'LINEAR':
            return ratings
        else:
            return super(AccuracyTipSelector, self).ratings_to_weight(ratings, alpha=self.settings[AccuracyTipSelectorSettings.ALPHA])

    def weighted_choice(self, approvers, weights):

        if self.settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] == 'MAXIMUM':
            # Instead of a weighted choice, always select the maximum.
            # If there is no unique maximum, choose the first one
            return approvers[weights.index(max(weights))]

        if self.settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] == 'WEIGHTED_CHOICE':
            return super(AccuracyTipSelector, self).weighted_choice(approvers, weights)
