from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self, purpose: str):
        self.purpose = purpose
