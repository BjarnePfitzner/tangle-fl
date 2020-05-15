import itertools
import random
import numpy as np

from .tip_selector import TipSelector

from baseline_constants import ACCURACY_KEY

# Adopted from https://docs.iota.org/docs/node-software/0.1/iri/references/iri-configuration-options
ALPHA = 0.001

class AccuracyTipSelector(TipSelector):
    def __init__(self, tangle, client):
        self.tangle = tangle

        # Build a map of transactions that directly approve a given transaction
        self.approving_transactions = {x: [] for x in self.tangle.transactions}
        for x, tx in self.tangle.transactions.items():
            for unique_parent in tx.parents:
                self.approving_transactions[unique_parent].append(x)

        self.ratings = self.compute_ratings(client)

    def compute_ratings(self, client):
        rating = {}
        original_params = client.model.get_params()

        for tx_id, tx in self.tangle.transactions.items():
            client.model.set_params(tx.load_weights())
            rating[tx_id] = client.test('train')[ACCURACY_KEY]
        
        client.model.set_params(original_params)

        return rating

    @staticmethod
    def ratings_to_weight(ratings):
        return ratings
    
    @staticmethod
    def weighted_choice(approvers, weights):
        # Instead of a weigthed choice, always select the maximum.
        # If there is no unique maximum, choose the first one

        return approvers[weights.index(max(weights))]
