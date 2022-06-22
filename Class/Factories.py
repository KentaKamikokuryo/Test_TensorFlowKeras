from Class.Interfaces import *
from Class.Model import *
from Class.DataSet import *

class ModelFactory():

    def __init__(self, hyper_model):

        self.hyper_model = hyper_model
        self.model_name = hyper_model["model_name"]

    def create(self) -> IModel:

        if self.model_name == Model.RF:

            model = RF(self.hyper_model)

        elif self.model_name == Model.LR:

            model = LR(self.hyper_model)

        elif self.model_name == Model.AB:

            model = AB(self.hyper_model)

        elif self.model_name == Model.GB:

            model = GB(self.hyper_model)

        elif self.model_name == Model.CB:

            model = CB(self.hyper_model)

        elif self.model_name == Model.XGB:

            model = XGB(self.hyper_model)

        elif self.model_name == Model.LGBM:

            model = LGBM(self.hyper_model)

        else:
            model = None

        return model

class DataSetFactory():

    def __init__(self, dataset_name: str):

        self.dataset_name = dataset_name

    def create(self) -> IDataSet:

        if self.dataset_name == DataSets.Iris:

            datasets = Iris(display=True)

        elif self.dataset_name == DataSets.Wine:

            datasets = Wine(display=True)

        elif self.dataset_name == DataSets.Digits:

            datasets = Digits(display=True)

        else:

            datasets = None

        return datasets
