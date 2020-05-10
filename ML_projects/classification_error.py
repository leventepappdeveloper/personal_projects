# Levente Papp
from __future__ import division, print_function
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib import pyplot as plt

# class 1
# np.seed = 7
np.random.seed(7)
train_data = np.random.normal(size = (100, 2))
train_labels = np.zeros(100)

# class 2
train_data = np.r_[train_data, np.random.normal(size = (100, 2), loc = 2)]
train_labels = np.r_[train_labels, np.ones(100)]

# check construct of rotation matrix
alpha = np.pi/2
Rotation = np.asarray([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])

# rotate data
rotated_data = np.matmul(train_data, Rotation)

from sklearn.tree import DecisionTreeClassifier

# function for the grid to use for the decision tree results visualization
def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

clf_tree = DecisionTreeClassifier( criterion = 'entropy', max_depth = 3, random_state = 17)

clf_tree.fit(train_data, train_labels)


xx, yy = get_grid(train_data)
predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy , predicted, cmap = 'autumn')
plt.scatter(train_data[:,0], train_data[:,1], c = train_labels, s = 100, cmap = 'autumn', edgecolors = 'black', linewidth = 1.5)


from sklearn.tree import DecisionTreeClassifier

# function for the grid to use for the decision tree results visualization
def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

clf_tree = DecisionTreeClassifier( criterion = 'entropy', max_depth = 3, random_state = 17)

clf_tree.fit(rotated_data, train_labels)


xx, yy = get_grid(rotated_data)
predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy , predicted, cmap = 'autumn')
plt.scatter(rotated_data[:,0], rotated_data[:,1], c = train_labels, s = 100, cmap = 'autumn', edgecolors = 'black', linewidth = 1.5)



# Plot the result for one rorate to check its working properly

# Levente Papp, 10/22/2019
# HW6, Problem 4
# Create test and train data
test_data_1 = train_data[80:100]
test_data_2 = train_data[180:]
train_data_1 = train_data[:80]
train_data_2 = train_data[100:180]
test_labels_1 = train_labels[80:100]
test_labels_2 = train_labels[180:]
train_labels_1 = train_labels[:80]
train_labels_2 = train_labels[100:180]

training_data = np.r_[train_data_1, train_data_2]
training_labels = np.r_[train_labels_1, train_labels_2]
testing_data = np.r_[test_data_1, test_data_2]
testing_labels = np.r_[test_labels_1, test_labels_2]

# Helper function to compute classification error
def classification_error(predicted, actual):
    error = 0
    for i in range(len(actual)):
        if predicted[i] != actual[i]:
            error = error + 1
    return float(error) / float(len(actual))

# For each angle between 0 and 90 degrees, we compute the error after rotating the whole data set (both training
# and testing data get rotates). Note that angles are in radians.
rad_to_degrees = 2*np.pi/360
alpha = 0
classification_errors = []
angles = []
while alpha < np.pi/2:
    Rotation = np.asarray([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
    rotated_training_data = np.matmul(training_data, Rotation)
    rotated_testing_data = np.matmul(testing_data, Rotation)
    
    clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth = 3, random_state=17)
    clf_tree.fit(rotated_training_data, training_labels)
    predictions = clf_tree.predict(rotated_testing_data)
    
    classification_errors.append(classification_error(predictions, testing_labels))
    angles.append(alpha)
    alpha = alpha + rad_to_degrees


# Plot above information: Classification error vs. rotation angle (in radians).
fig = plt.figure()
plt.plot(angles, classification_errors)
plt.xlabel("Rotation Angle (in radians)", fontsize = 14)
plt.ylabel("Classification Error", fontsize = 14)
plt.title("Decision Tree: Classification Error vs. Rotation Angle", fontsize = 18)


# From the Mean Squared Error, we can see that the Polynomial Regression model performs better than the Decision Tree
# Model (i.e. it has lower MSE value).
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

clf_tree = DecisionTreeRegressor()
clf_tree.fit(training_data, training_labels)
predictions = clf_tree.predict(testing_data)
error_dec_tree = mean_squared_error(predictions, testing_labels)
print("Decision Tree MSE: "  + str(error_dec_tree))

# Polynomial features (we add it to both the testing and training data)
poly = PolynomialFeatures(degree=3)
new_x = poly.fit_transform(training_data)
new_y = poly.fit_transform(training_labels.reshape(-1, 1))
new_x_test = poly.fit_transform(testing_data)
new_y_test = poly.fit_transform(testing_labels.reshape(-1, 1))

clf = linear_model.LinearRegression()
clf.fit(new_x, new_y)
predictions = clf.predict(new_x_test)
mse = mean_squared_error(predictions, new_y_test)
print("Polynomial Regression MSE: "  + str(mse))
