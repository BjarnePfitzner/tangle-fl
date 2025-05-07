import logging

import numpy as np

from tangle.core.tip_selection import TipSelector
from tangle.core.tip_selection.tip_selection_settings import TipSelectorSettings
from tangle.core.node import Node

# Adopted from https://docs.iota.org/docs/node-software/0.1/iri/references/iri-configuration-options


class AccuracyTipSelector(TipSelector):
    def __init__(self, tangle, tip_selection_settings, particle_settings=None):
        if particle_settings is None:
            particle_settings = {TipSelectorSettings.USE_PARTICLES: False}
        super().__init__(tangle, particle_settings=particle_settings)
        self.settings = tip_selection_settings

    def tip_selection(self, num_tips, node):
        if self.settings[TipSelectorSettings.SELECTION_STRATEGY] == "GLOBAL":
            self.tips.sort(key=lambda x: self.tx_rating(x, node), reverse=True)
            return self.tips[0:num_tips]
        else:
            return super(AccuracyTipSelector, self).tip_selection(num_tips, node)
    
    def _select_particle(self, particles, node):
        particle_ratings = [self.tx_rating(p, node) for p in particles]
        weights = self.ratings_to_weight(particle_ratings)
        return self.weighted_choice(particles, weights)

    def _compute_ratings(self, node: Node, tx=None):
        rating = {}

        txs = self._get_transactions_to_compute(tx)
        logging.debug(f"node {node.client_id} computes ratings for {len(txs)} transactions (tx={tx})")

        for tx_id in txs:
            rating[tx_id] = np.float64(node.test(node.tx_store.load_transaction_weights(tx_id), 'val')['accuracy'])

        # We (currently) do not care about the future-set-size-based rating
        # future_set_cache = {}
        # for tx in txs:
        #     rating[tx] *= len(TipSelector.future_set(tx, self.approving_transactions, future_set_cache)) + 1

        return rating

    def compute_ratings(self, node, tx=None):
        logging.debug(f"computing ratings for node {node.client_id}")
        rating = self._compute_ratings(node, tx)

        if self.settings[TipSelectorSettings.CUMULATE_RATINGS]:
            def cumulate_ratings(future_set, ratings):
                cumulated = 0
                for tx_id in future_set:
                    cumulated += ratings[tx_id]
                return cumulated

            # copy calculated accuracies
            accuracies = dict(rating)

            future_set_cache = {}
            for tx_id in rating:
                future_set = super().future_set(tx_id, self.approving_transactions, future_set_cache)
                rating[tx_id] = cumulate_ratings(future_set, accuracies) + accuracies[tx_id]

        # print("done computing ratings")
        self._update_ratings(node.client_id, rating)
    
    #### Provide template methods for subclasses (e.g. LazyAccuracyTipSelector)

    def _get_transactions_to_compute(self, tx):
        if self.settings[TipSelectorSettings.SELECTION_STRATEGY] == "GLOBAL" and not self.settings[TipSelectorSettings.CUMULATE_RATINGS]:
            return self.tips
        
        return self.tangle.transactions.keys()

    def _update_ratings(self, node_id, rating):
        self.ratings = rating

    #### Override weight functions with accuracy related settings
    def weighted_choice(self, approvers, weights):

        if self.settings[TipSelectorSettings.SELECT_FROM_WEIGHTS] == 'MAXIMUM':
            # Instead of a weighted choice, always select the maximum.
            # If there is no unique maximum, choose the first one
            return approvers[weights.index(max(weights))]

        if self.settings[TipSelectorSettings.SELECT_FROM_WEIGHTS] == 'WEIGHTED_CHOICE':
            return super(AccuracyTipSelector, self).weighted_choice(approvers, weights)
