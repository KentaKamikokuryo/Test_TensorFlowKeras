import numpy as np
import pandas as pd
from sklearn import datasets
from Class.Interfaces import IDataSet
from tabulate import tabulate

class DataSets:

    Iris = "iris"
    Wine = "wine"
    Digits = "digits"

class DataInfo:

    def __init__(self):

        self.data_names = [DataSets.Iris,
                           DataSets.Wine,
                           DataSets.Digits]

    def get_dataset_names(self):

        return self.data_names

class Iris(IDataSet):

    def __init__(self, display: bool=False):

        self.display = display

    def create(self):

        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)

        if self.display:

            print(df.head())

        X, y = datasets.load_iris(return_X_y=True)

        return X, y


class Wine(IDataSet):

    def __init__(self, display: bool = False):
        self.display = display

    def create(self):
        wine = datasets.load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)

        if self.display:
            print(df.head())

        X, y = datasets.load_wine(return_X_y=True)

        return X, y


class Digits(IDataSet):

    def __init__(self, display: bool = False):
        self.display = display

    def create(self):
        digits = datasets.load_digits
        df = pd.DataFrame(digits.data, columns=digits.feature_names)

        if self.display:
            print(df.head())

        X, y = datasets.load_digits(return_X_y=True)

        return X, y