from abc import ABC, abstractmethod


class BasePredictor(ABC):
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
