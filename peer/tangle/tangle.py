class Tangle:
    def __init__(self, genesis):
        self.transactions = {}
        self.genesis = genesis
        self.id = genesis.id
        self.add_transaction(genesis)

    def add_transaction(self, tip):
        self.transactions[tip.id] = tip
