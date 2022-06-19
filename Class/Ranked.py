import numpy as np
import pandas as pd
from tabulate import tabulate

class Ranked():

    hyperparameters_list_sorted: list
    mean_list_sorted: list
    std_list_sorted: list

    def __init__(self, model_name, path="", sort_type="ascending", metric_type="accuracy"):

        self.model_name = model_name
        self.path = path

        self.sort_type = sort_type
        self.metric_type = metric_type

        self.hyperparameters_list = []
        self.mean_list = []
        self.std_list = []

        self.name = "Model_" + str(self.model_name)

    def add(self, hyperparameter, mean=0, std=0):

        self.hyperparameters_list.append(hyperparameter)
        self.mean_list.append(mean)
        self.std_list.append(std)

    def ranked(self, display: bool = True, save: bool = True):

        if self.metric_type == "ascending":
            idx = np.argsort(self.mean_list)[::-1]
        elif self.metric_type == "descending":
            idx = np.argsort(self.mean_list)
        else:
            idx = np.argsort(self.mean_list)[::-1]

        self.hyperparameters_list_sorted = np.array(self.hyperparameters_list)[idx].tolist()
        self.mean_list_sorted = np.array(self.mean_list)[idx].tolist()
        self.std_list_sorted = np.array(self.std_list)[idx].tolist()

        print("Hyperparameters_list has been ranked")

        self.hyperparameter_best = self.hyperparameters_list_sorted[0]



        df = pd.DataFrame.from_dict(self.hyperparameters_list_sorted)
        df["mean"] = self.mean_list_sorted
        df["std"] = self.std_list_sorted

        if display:

            print(tabulate(df, headers="keys", tablefmt="psql"))

        if save:

            name = "ranked_" + self.name + ".csv"
            df.to_csv(self.path + name)
            print("Hyperparameters_list_sorted has been saved to: " + self.path + name)

    def save_best_hyperparameter(self):

        name = self.name + "_best.npy"

        np.save(self.path + name, self.hyperparameter_best)
        print("Hyperparameter_best has been saved to: " + self.path + name)

    def load_best_hyperparameter(self):

        name = self.name + "_best.npy"

        self.hyperparameter_best = np.load(self.path + name, allow_pickle=True).item()
        print("Hyperparameter_best has been loaded from: " + self.path + name)




