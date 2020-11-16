"""Interfaces for ClientModel """

from abc import ABC, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def set_params(self, model_params):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def train(self, data, num_epochs=1, batch_size=10):
        pass

    @abstractmethod
    def test(self, data):
        pass
