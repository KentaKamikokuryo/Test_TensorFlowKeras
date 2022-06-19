from abc import ABC, abstractmethod, abstractproperty
from typing import List
import numpy as np

class IModel(ABC):

    @abstractmethod
    def create(self):
        pass

class IDataSet(ABC):

    @abstractmethod
    def create(self):
        pass