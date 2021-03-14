from abc import ABC, abstractmethod

class TransactionStore(ABC):
    @abstractmethod
    async def load_transaction_weights(self, tx_id):
        pass

    @abstractmethod
    async def compute_transaction_id(self, tx):
        pass

    @abstractmethod
    async def save(self, tx, tx_weights):
        pass
