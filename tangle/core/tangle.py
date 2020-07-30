class Tangle:
    def __init__(self, transactions, genesis):
        self.transactions = transactions
        self.genesis = genesis

    def add_transaction(self, tip):
        self.transactions[tip.id] = tip
