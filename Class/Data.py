import numpy as np
from sklearn.model_selection import KFold, train_test_split


"""
data_info = {"test_size": 0.2, 
             "n_K_fold": 4,
             "k_shuffle": True}
"""

class Data():

    def __init__(self, data_info: dict):

        self.test_size = data_info["test_size"]
        self.n_K_fold = data_info["n_K_fold"]
        self.k_shuffle = data_info["k_shuffle"]

    def split_data(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)

        return X_train, X_test, y_train, y_test

    def k_split_data(self, X_train, y_train):

        kf = KFold(n_splits=self.n_K_fold, shuffle=self.k_shuffle)

        n = 0

        ks_X_fit = {}
        ks_X_valid = {}
        ks_y_fit = {}
        ks_y_valid = {}

        for fit_index, valid_index in kf.split(X=X_train, y=y_train):

            print("fit_index: ", fit_index, "valid_index: ", valid_index)

            X_fit, X_valid = X_train[fit_index], X_train[valid_index]
            y_fit, y_valid = y_train[fit_index], y_train[valid_index]

            ks_X_fit[str(n)] = X_fit
            ks_X_valid[str(n)] = X_valid
            ks_y_fit[str(n)] = y_fit
            ks_y_valid[str(n)] = y_valid

        return ks_X_fit, ks_X_valid, ks_y_fit, ks_y_valid




