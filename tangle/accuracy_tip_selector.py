from enum import Enum
import itertools
import random
import numpy as np

from .tip_selector import TipSelector

from baseline_constants import ACCURACY_KEY

# Adopted from https://docs.iota.org/docs/node-software/0.1/iri/references/iri-configuration-options

class AccuracyTipSelectorSettings(Enum):
    SELECTION_STRATEGY = 0
    CUMULATE_RATINGS = 1
    RATINGS_TO_WEIGHT = 2
    ALPHA = 3
    SELECT_FROM_WEIGHTS = 4

class AccuracyTipSelector(TipSelector):
    def __init__(self, tangle, client, settings):
        self.tangle = tangle
        self.settings = settings

        # Build a map of transactions that directly approve a given transaction
        self.approving_transactions = {x: [] for x in self.tangle.transactions}
        for x, tx in self.tangle.transactions.items():
            for unique_parent in tx.parents:
                self.approving_transactions[unique_parent].append(x)
        
        self.tips = []
        for x, tx in self.tangle.transactions.items():
            if len(self.approving_transactions[x]) == 0:
                self.tips.append(x)

        self.ratings = self.compute_ratings(client)
    
    def tip_selection(self, num_tips):
        if self.settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] == "GLOBAL":
            self.tips.sort(key=lambda x: self.ratings[x], reverse=True)
            return self.tips[0:num_tips]
        else:
            return super(AccuracyTipSelector, self).tip_selection(num_tips)

    def compute_ratings(self, client):
        rating = {}
        original_params = client.model.get_params()

        for tx_id, tx in self.tangle.transactions.items():
            client.model.set_params(tx.load_weights())
            rating[tx_id] = client.test('train')[ACCURACY_KEY]

        if self.settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS]:
            def cumulate_ratings(future_set, ratings):
                cumulated = 0
                for tx_id in future_set:
                    cumulated += ratings[tx_id]
                return cumulated
            
            future_set_cache = {}
            for tx_id in self.tangle.transactions:
                future_set = self.future_set(tx_id, self.approving_transactions, future_set_cache)
                rating[tx_id] = cumulate_ratings(future_set, rating) + rating[tx_id]
        
        client.model.set_params(original_params)

        return rating

    def ratings_to_weight(self, ratings):
        if self.settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] == 'LINEAR':
            return ratings
        else:
            return super(AccuracyTipSelector, AccuracyTipSelector).ratings_to_weight(ratings,alpha=self.settings[AccuracyTipSelectorSettings.ALPHA])
    
    def weighted_choice(self, approvers, weights):
    
        if self.settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] == 'MAXIMUM':
            # Instead of a weigthed choice, always select the maximum.
            # If there is no unique maximum, choose the first one
            return approvers[weights.index(max(weights))]
    
        if self.settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] == 'WEIGHTED_CHOICE':
            return super(AccuracyTipSelector, AccuracyTipSelector).weighted_choice(approvers, weights)
