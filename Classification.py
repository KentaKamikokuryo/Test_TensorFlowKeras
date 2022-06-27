import numpy as np
import pandas as pd
from Class.DataSet import *
from Class.Model import ModelInfo
from Class.Factories import *
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics
from Class.Ranked import Ranked
from Class.Hyperparameters import Hyperparameters


class Manager:

    hyper_model_search: bool
    train_final_model: bool
    perform_analysis: bool
    save_model: bool
    save_results: bool

    ranked: Ranked

    def __init__(self, data_name: str, data_info: dict):

        self.data_name = data_name
        self.test_size = data_info["test_size"]
        self.n_K_fold = data_info["n_K_fold"]
        self.k_shuffle = data_info["k_shuffle"]

        modelInfo = ModelInfo()
        self.model_names = modelInfo.get_model_names()

        self.path_search = ""
        self.path_best = ""

        self.__set_data()

    def __create_ML_model(self, hyper_model) -> IModel:

        fac = ModelFactory(hyper_model=hyper_model)
        ML_model = fac.create()

        return ML_model

    def __set_hyper(self, model_name):

        self.ranked = Ranked(model_name=model_name,
                             path=self.path_search)

        if self.hyper_model_search:

            hyperParams = Hyperparameters(model_name=model_name)
            self.hyper_model_list, self.hyper_model_dict = hyperParams.generate_hypermodel(display=True)

        else:

            self.ranked.load_ranked_list(display=True)
            self.ranked.load_best_hyperparameter()
            self.hyper_model_best = self.ranked.hyperparameter_best.copy()
            self.hyper_model_list_sorted = self.ranked.hyperparameters_list_sorted.copy()

    def __set_data(self):

        data_fac = DataSetFactory(dataset_name=self.data_name)
        dataset_model = data_fac.create()

        # create data-sets
        self.X, self.y = dataset_model.create()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size)

    def __fit_valid(self, hyper_model):

        cv = KFold(n_splits=self.n_K_fold, shuffle=False)
        ML_model = self.__create_ML_model(hyper_model=hyper_model)

        scores = []

        for fit_index, valid_index in cv.split(X=self.X_train):

            # get train and test data
            X_fit, X_valid = self.X_train[fit_index], self.X_train[valid_index]
            y_fit, y_valid = self.y_train[fit_index], self.y_train[valid_index]

            # fit model
            ML_model.fit(X_fit, y_valid)

            # predict test data
            y_pred = ML_model.predict(X_fit)

            # loss
            metric = metrics.accuracy_score(y_true=y_valid, y_pred=y_pred)
            scores.append(metric)

        metric_mean = np.mean(scores)
        metric_std = np.std(scores)

        return metric_mean, metric_std

    def __test(self, hyper_model):

        ML_model = self.__create_ML_model(hyper_model=hyper_model)

        ML_model.fit(self.X_train, self.y_train)
        y_pred = ML_model.predict(self.X_test)

        accuracy = metrics.accuracy_score(y_true=self.y_test, y_pred=y_pred)

        return accuracy

    def __run_search(self):

        for model_name in self.model_names:

            self.__set_hyper(model_name=model_name)

            for hyper_model in self.hyper_model_list:

                metric_mean, metric_std = self.__fit_valid(hyper_model=hyper_model)
                self.ranked.add(hyperparameter=hyper_model, mean=metric_mean, std=metric_std)

            self.ranked.ranked(display=True, save=self.save_best_search)
            self.ranked.save_best_hyperparameter()

            print("Run search is done on " + str(self.data_name) + "with model " + model_name)

        print("Data-sets search is done on data-sets name: " + str(self.data_name))

    def __run_comparison(self):

        ranked_comparison = Ranked(model_name="Comparison_models",
                                   path="")

        for model_name in self.model_names:

            accuracy = self.__test(hyper_model=self.hyper_model_best)

            ranked_comparison.add(hyperparameter=self.hyper_model_best,
                                  mean=accuracy,
                                  std=0)

        ranked_comparison.ranked(display=True, save=self.save_best_comparison)
        ranked_comparison.save_best_hyperparameter()

    def set_interface(self, interface_dict: dict):

        self.hyper_model_search = interface_dict["hyper_model_search"]
        self.save_best_search = interface_dict["save_best_search"]
        self.save_best_comparison = interface_dict["save_best_comparison"]

    def run(self):

        if self.hyper_model_search:

            self.__run_search()
            self.__run_comparison()

        else:

            self.__run_comparison()


Is = {"search": True,
      "run": False}

for I in Is.keys():

    if I == "search":

        interface_dict = {"hyper_model_search": True,
                          "save_best_search": True,
                          "save_best_comparison": True}

    elif I == "run":

        interface_dict = {"hyper_model_search": False,
                          "save_best_search": True,
                          "save_best_comparison": True}

    else:

        interface_dict = {}

    dataInfo = DataInfo()
    data_names = dataInfo.get_dataset_names()
    data_info = dataInfo.get_data_info()

    for data_name in data_names:

        manager = Manager(data_name=data_name, data_info=data_info)
        manager.set_interface(interface_dict=interface_dict)
        manager.run()