"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.random import rand
import random
from .abstract_model import AbstractModel


class NoTfModel(AbstractModel):

    def __init__(self, seed):
        # Is close enough to weight_size of femnist model
        self._parameters = rand(3300000)

    def set_params(self, model_params):
        self._parameters = model_params

    def get_params(self):
        return self._parameters

    def train(self, data, num_epochs=1, batch_size=10):
        self._parameters = rand(3300000)
        return None

    def test(self, data):
        return {
            "loss": random.random(),
            "accuracy": random.random()
        }
