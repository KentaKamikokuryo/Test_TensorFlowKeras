from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics

dataset = datasets.load_digits()

# dividing the datasets into two parts. i.e. training data sets and test datasets
X, y = datasets.load_digits(return_X_y=True)

# Splitting arrays or matrices into random train and test subsets
# i.e. 70 % training data-sets and 30% test data-sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

print(df.head())

# creating a RF classifier
clf = CatBoostClassifier(n_estimators=100)

cv = KFold(n_splits=5, shuffle=True)

scores = []
for train_index, test_index in cv.split(X=X):

    # get train and test data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # fit model
    clf.fit(X_train, y_train)
    # predict test data
    y_pred = clf.predict(X_test)
    # loss
    metric = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    scores.append(metric)


# using metrics module for accuracy calculation
print("MEAN ACCURACY OF THE MODEL: " + str(np.mean(scores)) + " - STD: " + str(np.std(scores)))

# predicting which type of flower it is
# clf.predict([[3, 3, 2, 2]])

# using the feature importance variable
feature_imp = pd.Series(clf.feature_importances_, index=dataset.feature_names).sort_values(ascending=False)
print(feature_imp)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=1.5)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Importance Features")
plt.show()

# # Tuning random forest
# # Number of trees in random forest
# n_estimators = np.linspace(100, 3000, int((3000 - 100) / 200) + 1, dtype=int)
#
# # Number of features to consider at every split
# max_features = ["auto", "sqrt"]
#
# # Maximum number of levels in trees
# max_depth = [1, 5, 10, 20, 50, 75, 100, 150, 200]
#
# # Minimum number of samples required to split a node
# min_samples_split = [1, 2, 5, 10, 15, 20, 30]
#
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 3, 4]
#
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
#
# # Criterion
# criterion = ["gini", "entropy"]
#
# random_grid = {'n_estimators': n_estimators,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap,
#                'criterion': criterion}
#
# clf_ramdom = RandomizedSearchCV(estimator=clf,
#                                 param_distributions=random_grid,
#                                 n_iter=30,
#                                 cv=5,
#                                 verbose=2,
#                                 random_state=42, n_jobs=4)
#
# clf_ramdom.fit(X_train, y_train)
# print("BEST PARAMS", clf_ramdom.best_params_)



