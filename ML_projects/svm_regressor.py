# Levente Papp, 10/15/2019
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('abalone.txt', header=None)
print( data.head())
print (data.dtypes)
print(np.shape(data))

# encode categorical data
le = preprocessing.LabelEncoder()
le.fit(data.iloc[:,0])
data.iloc[:,0] = le.transform(data.iloc[:,0])
print(le.classes_)
print(np.shape(data))
X = np.asarray(data.iloc[:,0:8])
print(X[0:5])
print(np.shape(X))
y = np.asarray(data.iloc[:,8]).astype(float)
print(y[0:5])

# This is my SVM regressor code. 
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error 

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 47)

# train model on test set
model = SVR(kernel="linear")
model.fit(X_train, y_train)

# test on test data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error: " + str(mse))



