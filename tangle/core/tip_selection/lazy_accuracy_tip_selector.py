from enum import Enum

from .tip_selector import TipSelector

from ...models.baseline_constants import ACCURACY_KEY

# Adopted from https://docs.iota.org/docs/node-software/0.1/iri/references/iri-configuration-options

class AccuracyTipSelectorSettings(Enum):
    SELECTION_STRATEGY = 0
    CUMULATE_RATINGS = 1
    RATINGS_TO_WEIGHT = 2
    ALPHA = 3
    SELECT_FROM_WEIGHTS = 4

class LazyAccuracyTipSelector(TipSelector):
    def __init__(self, tangle, settings):
        super().__init__(tangle)
        self.settings = settings
        self.accuracies = {}
        # assert(self.settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] != "GLOBAL")
        # assert(self.settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS] == False)
        # assert(self.settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] != 'LINEAR')
        # print(self.settings)

    def tx_rating(self, tx, node):
        if tx not in self.accuracies:
            self.accuracies[tx] = node.test(node.tx_store.load_transaction_weights(tx), 'train')[ACCURACY_KEY]

        accuracy = self.accuracies[tx]
        return self.ratings[tx] * accuracy

    def compute_ratings(self, node):
        super().compute_ratings(node)

    def ratings_to_weight(self, ratings, alpha=None):
        return super().ratings_to_weight(ratings, alpha=0.1)

    def weighted_choice(self, approvers, weights):
        return super().weighted_choice(approvers, weights)
