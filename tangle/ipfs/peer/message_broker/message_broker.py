from abc import ABC, abstractmethod


class MessageBroker(ABC):
    pass

    @abstractmethod
    async def subscribe(self):
        pass

    @abstractmethod
    def publish(self, tx):
        pass
