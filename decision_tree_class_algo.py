# 16 April 2025
# Ensamble learning- Bagging
# Decision Tree Classificaion Algorithm

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\GauravKunal\Desktop\DS\Machine Learning\#2 Classification\#5 Decision Tree\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Tree algo never required feature scale
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

# Traning the Decision Tree Classification model on the training set
from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier() # ac-0.9125
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=None) # ac-0.925
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance






'''
Further task is use validation data(vehicle purchase or not) 
and apply all classification algorithm to that validation data(future data)
and specific algo gives more accuracy do the deployment of that algo.


Vehical price pridiction

logit gives ac - 92.50
svm - ac-95
knn - ac-95
naive - ac-90
dt - ac-92.50


Do hyperparameter tuning to all above algo & check which algo gives
high ac and do deployment with that algo.
'''








