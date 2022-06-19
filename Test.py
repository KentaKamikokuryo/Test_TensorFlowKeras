from sklearn import datasets

iris = datasets.load_iris()
wine = datasets.load_wine()
digits = datasets.load_digits()

X_iris, y_iris = datasets.load_iris(return_X_y=True)
X_wine, y_wine = datasets.load_wine(return_X_y=True)
X_digits, y_digits = datasets.load_digits(return_X_y=True)