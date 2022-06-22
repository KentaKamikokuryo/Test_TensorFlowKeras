import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from Class.Factories import ModelFactory

from Class.Interfaces import IModel


class Model:

    RF = "RandomForest"
    LR = "LogisticRegression"
    AB = "AdaBoost"
    GB = "GradientBoost"
    CB = "CatBoost"
    XGB = "XGBoost"
    LGBM = "LightGBM"

class ModelInfo:

    def __init__(self):
        self.model_names = [Model.RF,
                            Model.LR,
                            Model.AB,
                            Model.GB,
                            Model.CB,
                            Model.XGB,
                            Model.LGBM]

    def get_model_names(self):
        return self.model_names


class RF(IModel):

    def __init__(self, hyper: dict):

        self.hyper = hyper

        # Number of trees in random forest
        self.n_estimators = hyper["n_estimators"]

        # Minimum number of samples required to split a node
        self.min_samples_leaf = hyper["min_samples_leaf"]

        # Criterion
        self.criterion = hyper["criterion"]

    def create(self):

        model = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, min_samples_leaf=self.min_samples_leaf)

        return model

class LR(IModel):

    def __init__(self, hyper: dict):

        pass

    def create(self):

        model = LogisticRegression()

        return model

class AB(IModel):

    def __init__(self, hyper: dict):

        pass

    def create(self):

        model = AdaBoostClassifier()

        return model

class GB(IModel):

    def __init__(self, hyper: dict):

        pass

    def create(self):

        model = GradientBoostingClassifier()

        return model

class XGB(IModel):

    def __init__(self, hyper: dict):

        pass

    def create(self):

        model = XGBClassifier()

        return model

class LGBM(IModel):

    def __init__(self, hyper: dict):

        pass

    def create(self):

        model = LGBMClassifier()

        return model

class CB(IModel):

    def __init__(self, hyper: dict):

        pass

    def create(self):

        model = CatBoostClassifier()

        return model


class SupervisedModel():

    model_fac: ModelFactory
    model: IModel

    def __init__(self, hyper_model: dict):

        self.hyper_model = hyper_model
        self.model_name = self.hyper_model["model_name"]

        self.create_model()

    def create_model(self):

        self.model_fac = ModelFactory(hyper_model=self.hyper_model)
        self.model = self.model_fac.create()





