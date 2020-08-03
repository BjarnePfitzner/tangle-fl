class Tangle:
    def __init__(self, transactions, genesis):
        self.transactions = transactions
        self.genesis = genesis
        self.tips = self.find_tips(transactions)

    def find_tips(self, transactions):
        potential_tips = set(transactions.keys())
        for _, tx in transactions.items():
            for parent_tx in tx.parents:
                potential_tips.discard(parent_tx)
        return potential_tips

    def add_transaction(self, tip):
        self.transactions[tip.id] = tip
        for parent_tx in tip.parents:
            self.tips.discard(parent_tx)
        self.tips.add(tip.id)
