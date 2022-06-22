import numpy as np
import pandas as pd
import math, itertools, os
from tabulate import tabulate
from Class.Model import Model

class Hyperparameters():

    def __init__(self, model_name):

        self.model_name = model_name

    def generate_hypermodel(self, display=False):

        grid = {}

        if self.model_name == Model.RF:

            grid = {"n_estimators": [10, 50, 100, 200, 300],
                    "min_samples_leaf": [1, 2, 3],
                    "criterion": ["gini", "entropy"],
                    "model_name": [self.model_name]}

        elif self.model_name == Model.LR:

            grid = {"model_name": [self.model_name]}

        elif self.model_name == Model.AB:

            grid = {"model_name": [self.model_name]}

        elif self.model_name == Model.GB:

            grid = {"model_name": [self.model_name]}

        elif self.model_name == Model.CB:

            grid = {"model_name": [self.model_name]}

        elif self.model_name == Model.XGB:

            grid = {"n_estimators": [10, 50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3,6,10],
                    "model_name": [self.model_name]}

        elif self.model_name == Model.LGBM:

            grid = {"model_name": [self.model_name]}

        else:

            grid = dict()

        keys, values = zip(*grid.items())
        grid_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame.from_dict(grid_combination)

        if display:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return grid_combination, grid

