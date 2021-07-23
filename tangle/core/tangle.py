import itertools
import random
import numpy as np

class Tangle:
    def __init__(self, transactions, genesis, unresolved_parents = []):
        self.transactions = transactions
        self.genesis = genesis
        self.tips = self.find_tips(transactions)
        self.unresolved_parents = set(unresolved_parents)
        # self.depth_cache = self.calculate_depth()

    # def calculate_depth(self):
    #     """
    #     Returns dict with {depth: [transaction_ids]}

    #     Depth is defined as the length of the longest directed path from a tip to the transaction.
    #     """

    #     # first calculate depth for each transaction

    #     depth_per_transaction = {}
    #     depth = 0
    #     current_transactions = [tx for _, tx in self.transactions.items() if tx.id in self.tips]

    #     for t in current_transactions:
    #         depth_per_transaction[t.id] = depth

    #     depth += 1
    #     parents = set(itertools.chain(*[tx.parents for tx in current_transactions]))

    #     while len(parents) > 0:
    #         for parent in parents:
    #             depth_per_transaction[parent] = depth

    #         depth += 1
    #         current_transactions = [tx for _, tx in self.transactions.items() if tx.id in parents]
    #         parents = set(itertools.chain(*[tx.parents for tx in current_transactions]))

    #     # build desired dict structure

    #     transactions_per_depth = {}

    #     for d in range(depth):
    #         transactions_per_depth[d] = [tx for tx, tx_depth in depth_per_transaction.items() if tx_depth == d]

    #     return transactions_per_depth

    def find_tips(self, transactions):
        potential_tips = set(transactions.keys())
        for _, tx in transactions.items():
            for parent_tx in tx.parents:
                potential_tips.discard(parent_tx)
        return potential_tips

    def add_transaction(self, tx):
        if tx.id in self.unresolved_parents:
            self.unresolved_parents.remove(tx.id)
        elif tx.id not in self.transactions:
            self.tips.add(tx.id)

        self.transactions[tx.id] = tx

        for parent_tx in tx.parents:
            self.tips.discard(parent_tx)

    def get_transaction_ids_of_depth_interval(self, depth_start, depth_end):
        """
        Returns all transaction ids from the tangle, which have a depth between or equal depth_start and depth_end.
        """
        gathered_transaction_ids = []

        for depth in range(depth_start, depth_end + 1):
            if depth in self.depth_cache:
                gathered_transaction_ids.extend(self.depth_cache[depth])

        # If no transaction was found inside this interval return genesis
        if len(gathered_transaction_ids) == 0:
            gathered_transaction_ids.append(self.genesis)

        return gathered_transaction_ids

    def choose_trunk_branch(self):
        sample_size = 30
        sample_depth = 20

        sampled_tips = set()
        sample_depths = {}
        sample_hits = {}

        tips = list(self.tips)

        for i in range(sample_size):
            current_tx = random.choice(tips)
            for d in range(sample_depth):
                if current_tx not in sampled_tips:
                    sampled_tips.add(current_tx)
                    sample_depths[current_tx] = d
                    sample_hits[current_tx] = 1
                else:
                    sample_depths[current_tx] = max(sample_depths[current_tx], d)
                    sample_hits[current_tx] = sample_hits[current_tx] + 1

                parents = [p for p in self.transactions[current_tx].parents if p in self.transactions]
                if len(parents) == 0:
                    break

                current_tx = random.choice(parents)

        b = sum(map(lambda r: np.exp(0.001 * sample_depths[r]), sampled_tips))
        # Normalizing depth is a little nonsense since it's limit to sample_depth. Rather normalize sample_hits then (max: 50)
        trunk_scores  = {t: (np.exp(sample_depths[t] * 0.001) / b) * sample_hits[t] for t in sampled_tips}
        branch_scores = {t: (np.exp(sample_depths[t] * 0.001) / b) * sample_hits[t] for t in sampled_tips}

        trunk = max(trunk_scores, key=lambda key: trunk_scores[key])
        branch = max(branch_scores, key=lambda key: branch_scores[key])

        return trunk, branch
