from abc import ABC, abstractmethod


class MessageBroker(ABC):
    pass

    @abstractmethod
    async def subscribe(self, tangle_id: str):
        pass

    @abstractmethod
    def publish(self, tangle_id: str, value: dict):
        pass
