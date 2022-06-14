import pandas as pd
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cd = os.getcwd()
data_pass = cd + "//archive//ks-projects-201801.csv"

df = pd.read_csv(data_pass)
df_shape = df.shape
df.head()

df["deadline"] = pd.to_datetime(df["deadline"])
df["launched"] = pd.to_datetime(df["launched"])
df["days"] = (df["deadline"] - df["launched"]).dt.days

df = df[(df["state"] == "successful") | (df["state"] == "failed")]
df["state"] = df["state"].replace("failed", 0)
df["state"] = df["state"].replace("successful", 1)

df = df.drop(["ID","name","deadline","launched","backers","pledged","usd pledged","usd_pledged_real","usd_goal_real"], axis=1)
df = pd.get_dummies(df,drop_first = True)

train_data = df.drop("state", axis=1)
y = df["state"].values
X = train_data.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

clf = RandomForestClassifier(random_state=1234)
clf.fit(X_train, y_train)
print("score=", clf.score(X_test, y_test))