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

    def get_transaction_ids_of_time_interval(self, steps_back, width):
        """
        Returns all transaction ids from the tangle, which are in the specified interval.

        First the end of the interval ist computed by going n steps back from the last generation in the tangle.
        Next, the interval starts `width` timesteps before the end.

        E.g. current latest generation = 50, steps_back = 5, width = 10: interval = [35, 45]
        """
        gathered_transaction_ids = []

        tip_transactions = [self.transactions[x] for x in self.tips]
        last_generation = max([n.metadata['time'] for n in tip_transactions])
        end = max(0, last_generation - steps_back)
        start = max(0, end - width)

        for t_id in self.transactions:
            if "time" in self.transactions[t_id].metadata:
                issued_time = self.transactions[t_id].metadata["time"]
                if issued_time >= start and (end is None or issued_time <= end):
                    gathered_transaction_ids.append(t_id)
            elif start == 0:
                gathered_transaction_ids.append(t_id)
        
        return gathered_transaction_ids
