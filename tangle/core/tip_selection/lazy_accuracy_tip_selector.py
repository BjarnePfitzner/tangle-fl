import time

from tangle.core.tip_selection.tip_selector import TipSelector
from .accuracy_tip_selector import AccuracyTipSelectorSettings

from ...models.baseline_constants import ACCURACY_KEY

class LazyAccuracyTipSelector(TipSelector):
    def __init__(self, tangle, settings, particle_settings):
        super().__init__(tangle, particle_settings=particle_settings)
        self.settings = settings
        self.accuracies = {}
        assert(self.settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] == 'WALK')
        assert(self.settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS] == False)
        assert(self.settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] == 'ALPHA')
        assert(self.settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] == 'WEIGHTED_CHOICE')

    def tx_rating(self, tx, node):
        if tx not in self.accuracies:
            # begin = time.time()
            # print(f'evaluating for node {node.id} tx {tx}')
            self.accuracies[tx] = node.test(node.tx_store.load_transaction_weights(tx), 'train')[ACCURACY_KEY]
            # print(f'done in {time.time() - begin}')

        return self.accuracies[tx]

    def compute_ratings(self, node):
        # We (currently) do not care about the future-set-size-based rating
        self.ratings = {}
