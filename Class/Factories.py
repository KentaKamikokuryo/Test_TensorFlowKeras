from Class.Interfaces import *
from Class.Model import *
from Class.DataSet import *

class ModelFactory():

    def __init__(self, hyper_model):

        self.hyper_model = hyper_model
        self.model_name = hyper_model["model_name"]

    def create(self) -> IModel:

        if self.model_name == "RF":

            model = RF(self.hyper_model)

        elif self.model_name == "NV":

            model = NaiveBayes(self.hyper_model)

        else:
            model = None

        return model

class DataSetFactory():

    def __init__(self, data_info: dict):

        self.dataset_name = data_info["dataset_name"]

    def create(self) -> IDataSet:

        if self.dataset_name == "iris":

            datasets = Iris(display=True)

        elif self.dataset_name == "wine":

            datasets = Wine(display=True)

        elif self.dataset_name == "digits":

            datasets = Digits(display=True)

        else:

            datasets = None

        return datasets



