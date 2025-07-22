import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


info = pd.read_csv("Titanic-Dataset.csv")

info

info.info()

df = info[~info["Survived"].isnull()]

df = info[~info["Age"].isnull()]

df = info[~info["Cabin"].isnull()]

df = info[~info["Fare"].isnull()]

df.ffill(inplace =True)

df.describe().sum()

df.isnull().sum()

df.describe()

plt.boxplot(df["SibSp"])
plt.show()

df.drop(columns=["Cabin"],inplace=True)

df

df.select_dtypes(int,float).columns

plt.boxplot(df["Age"])
plt.show()

plt.boxplot(df["Fare"])
plt.show()

def IQR (x):
  q1 = x.quantile(0.25)
  q3 = x.quantile(0.75)
  IQR = q3-q1
  upper_bound = q3 + 1.5*IQR
  lower_bound = q1 - 1.5*IQR
  print("lower_bound",lower_bound)
  print("upper_bound",upper_bound)

IQR(df["Fare"])

IQR(df["Age"])

IQR(df["SibSp"])

IQR(df["Pclass"])



df["Fare"] = np.where(df["Fare"]>65.6344,65.6344,df["Fare"])

df["Pclass"] = np.where(df["Pclass"]>4.5,4.5,df["Pclass"])

df["SibSp"] = np.where(df["SibSp"]>2.5,2.5,df["SibSp"])

df["Age"] = np.where(df["Age"]>64.81,math.floor(64.81),df["Age"])

plt.boxplot(df["SibSp"])
plt.show()

X = df[['PassengerId','Pclass','Age','SibSp','Fare','Parch']]
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print(knn.score(X_train,y_train))
print(knn.score(X_test,y_test))

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_sc,y_train)
print("Train",knn.score(X_train_sc,y_train))
print("Test",knn.score(X_test_sc,y_test))

train_acc = []
test_acc = []
for i in range(1,50,2):
  knn = KNeighborsClassifier(n_neighbors = i)
  knn.fit(X_train_sc,y_train)
  train_acc.append(knn.score(X_train_sc,y_train))
  test_acc.append(knn.score(X_test_sc,y_test))

plt.figure(figsize = (10,5))
plt.plot(train_acc,"bo--", label = "Train")
plt.plot(test_acc,"go--", label = "Test")
plt.legend()
plt.show()

knn = KNeighborsClassifier(n_neighbors = 37)
knn.fit(X_train_sc,y_train)
print("Train",knn.score(X_train_sc,y_train))
print("Test",knn.score(X_test_sc,y_test))

print("train", knn.score(X_train_sc,y_train))
print("test", knn.score(X_test_sc,y_test))

