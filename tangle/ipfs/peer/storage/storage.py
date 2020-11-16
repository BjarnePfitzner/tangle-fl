from abc import ABC, abstractmethod


class Storage(ABC):
    pass

    # Are used to store & retrieve weights
    # Returns the file path or Hash
    @abstractmethod
    def add_file(self, content):
        pass

    @abstractmethod
    def get_file(self, path: str):
        pass

    # Are used to store transactions and to retrieve missed transactions
    # Returns the file path or Hash
    @abstractmethod
    def add_json(self, value: dict):
        pass

    @abstractmethod
    def get_json(self, path: str):
        pass
