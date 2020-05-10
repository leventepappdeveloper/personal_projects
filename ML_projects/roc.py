# Levente Papp, 10/15/2019
# 
# After reading in the data, I create an extra column in each dataset than has a value of 0 if num <= 2, else 1. 
# This way, I can plot ROC curves and compute AUC values by making it a binary classification problem.
# As a result, for each dataset, I create two seperate models: a) using the original 5 classes, I output a 
# confusion matrix, b) using the 2 "combined" classes, I compute AUC values and draw ROC curves. Please see
# my model comparisons below.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

cleveland_heart = pd.read_csv("cleveland-heart-edited.txt", header = None)
long_beach_heart = pd.read_csv("full-long-beach-heart-edited.txt", header = None)
switzerland_heart = pd.read_csv("full-switzerland-heart-edited.txt", header = None)
hungarian_heart = pd.read_csv("hungarian-heart-edited.txt", header = None)

# Subset columns
cols_to_keep = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]
long_beach_heart = long_beach_heart[cols_to_keep]
switzerland_heart = switzerland_heart[cols_to_keep]

# Column names
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
               "ca", "thal", "num"]
cleveland_heart.columns = column_names
long_beach_heart.columns = column_names
switzerland_heart.columns = column_names
hungarian_heart.columns = column_names

# helper columns so we only have 2 classes for ROC (group nums <= 2 into 1 group):
cleveland_heart['helper'] = np.where(cleveland_heart['num']<=2, 0, 1)
long_beach_heart['helper'] = np.where(long_beach_heart['num']<=2, 0, 1)
switzerland_heart['helper'] = np.where(switzerland_heart['num']<=2, 0, 1)
hungarian_heart['helper'] = np.where(hungarian_heart['num']<=2, 0, 1)

# Cleveland DATA
X_cleveland = cleveland_heart[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
                                     "oldpeak", "slope", "ca", "thal"]]
y_cleveland = cleveland_heart[["num"]]
y_cleveland_2 = cleveland_heart[["helper"]]

# Long Beach DATA
X_long_beach = long_beach_heart[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
                                       "oldpeak", "slope", "ca", "thal"]]
y_long_beach = long_beach_heart[["num"]]
y_long_beach_2 = long_beach_heart[["helper"]]

# Switzerland DATA
X_switzerland = switzerland_heart[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
               "ca", "thal"]]
y_switzerland = switzerland_heart[["num"]]
y_switzerland_2 = switzerland_heart[["helper"]]

# Hungary DATA
X_hungary = hungarian_heart[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
               "ca", "thal"]] 
y_hungary = hungarian_heart[["num"]]
y_hungary_2 = hungarian_heart[["helper"]]


# Train-test splits (1 classifies into 5 groups, the other into 2 groupss)
X_cleveland_train_1, X_cleveland_test_1, y_cleveland_train_1, y_cleveland_test_1 = train_test_split(
    X_cleveland, y_cleveland, test_size = 0.2, random_state = 47)
X_cleveland_train_2, X_cleveland_test_2, y_cleveland_train_2, y_cleveland_test_2 = train_test_split(
    X_cleveland, y_cleveland_2, test_size = 0.2, random_state = 47)

X_long_beach_train_1, X_long_beach_test_1, y_long_beach_train_1, y_long_beach_test_1 = train_test_split(
    X_long_beach, y_long_beach, test_size = 0.2, random_state = 47)
X_long_beach_train_2, X_long_beach_test_2, y_long_beach_train_2, y_long_beach_test_2 = train_test_split(
    X_long_beach, y_long_beach_2, test_size = 0.2, random_state = 47)

X_switzerland_train_1, X_switzerland_test_1, y_switzerland_train_1, y_switzerland_test_1 = train_test_split(
    X_switzerland, y_switzerland, test_size = 0.2, random_state = 47)
X_switzerland_train_2, X_switzerland_test_2, y_switzerland_train_2, y_switzerland_test_2 = train_test_split(
    X_switzerland, y_switzerland_2, test_size = 0.2, random_state = 47)

X_hungary_train_1, X_hungary_test_1, y_hungary_train_1, y_hungary_test_1 = train_test_split(
    X_hungary, y_hungary, test_size = 0.2, random_state = 47)
X_hungary_train_2, X_hungary_test_2, y_hungary_train_2, y_hungary_test_2 = train_test_split(
    X_hungary, y_hungary_2, test_size = 0.2, random_state = 47)

# HELPER FUNCTIONS
def roc_plotter(model_2, X_data, y_data_2):
    predicted_prob = model_2.predict_proba(X_data)
    predicted_prob = predicted_prob[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_data_2, predicted_prob, pos_label=1)
    fpr = np.insert(fpr, 0, 0)
    tpr = np.insert(tpr, 0, 0)
    lw = 2
    print("AUC: " + str(metrics.roc_auc_score(y_data_2, predicted_prob)))
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()
    
def print_confusion_matrix(model_1, X_data, y_data):
    predictions = model_1.predict(X_data)
    print(metrics.confusion_matrix(y_data, predictions))
    
def run(classifier, X_train_1, X_test_1, y_train_1, y_test_1, X_train_2, X_test_2, y_train_2, y_test_2):
    m_1 = classifier.fit(X_train_1, y_train_1)
    predictions = m_1.predict(X_test_1)
    print(metrics.confusion_matrix(y_test_1, predictions))
    
    m_2 = classifier.fit(X_train_2, y_train_2)
    roc_plotter(m_2, X_test_2, y_test_2)


# ExtraTreesClassifier Model
# As we can see based on the confusion matrices and the ROC/AUC values, this model is a reasonably good predictor
# for the CLEVELAND dataset, however, it performs worse than a random classifier for the SWITZERLAND dataset. 
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

print("CLEVELAND")
run(ExtraTreesClassifier(), X_cleveland_train_1, X_cleveland_test_1, y_cleveland_train_1, y_cleveland_test_1,
   X_cleveland_train_2, X_cleveland_test_2, y_cleveland_train_2, y_cleveland_test_2)

print("LONG BEACH")
run(ExtraTreesClassifier(), X_long_beach_train_1, X_long_beach_test_1, y_long_beach_train_1, y_long_beach_test_1,
   X_long_beach_train_2, X_long_beach_test_2, y_long_beach_train_2, y_long_beach_test_2)

print("SWITZERLAND")
run(ExtraTreesClassifier(), X_switzerland_train_1, X_switzerland_test_1, y_switzerland_train_1, y_switzerland_test_1,
   X_switzerland_train_2, X_switzerland_test_2, y_switzerland_train_2, y_switzerland_test_2)

print("HUNGARY")
run(ExtraTreesClassifier(), X_hungary_train_1, X_hungary_test_1, y_hungary_train_1, y_hungary_test_1,
   X_hungary_train_2, X_hungary_test_2, y_hungary_train_2, y_hungary_test_2)

# Logistic Regression Model
# This model seems to outperform the above ExtraTreesClassifier model for each dataset. It performs especially well
# on the Hungarian dataset, with an AUC of 0.91. 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

print("CLEVELAND")
run(LogisticRegression(), X_cleveland_train_1, X_cleveland_test_1, y_cleveland_train_1, y_cleveland_test_1,
   X_cleveland_train_2, X_cleveland_test_2, y_cleveland_train_2, y_cleveland_test_2)

print("LONG BEACH")
run(LogisticRegression(), X_long_beach_train_1, X_long_beach_test_1, y_long_beach_train_1, y_long_beach_test_1,
   X_long_beach_train_2, X_long_beach_test_2, y_long_beach_train_2, y_long_beach_test_2)

print("SWITZERLAND")
run(LogisticRegression(), X_switzerland_train_1, X_switzerland_test_1, y_switzerland_train_1, y_switzerland_test_1,
   X_switzerland_train_2, X_switzerland_test_2, y_switzerland_train_2, y_switzerland_test_2)

print("HUNGARY")
run(LogisticRegression(), X_hungary_train_1, X_hungary_test_1, y_hungary_train_1, y_hungary_test_1,
   X_hungary_train_2, X_hungary_test_2, y_hungary_train_2, y_hungary_test_2)

# Gaussian Model
# Again this model does very well on the Hungarian dataset, but performs rather poorly on the Switzerland data. In 
# fact, its AUC of 0.47 makes it a worse-than-random classifier. 
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

print("CLEVELAND")
run(GaussianNB(), X_cleveland_train_1, X_cleveland_test_1, y_cleveland_train_1, y_cleveland_test_1,
   X_cleveland_train_2, X_cleveland_test_2, y_cleveland_train_2, y_cleveland_test_2)

print("LONG BEACH")
run(GaussianNB(), X_long_beach_train_1, X_long_beach_test_1, y_long_beach_train_1, y_long_beach_test_1,
   X_long_beach_train_2, X_long_beach_test_2, y_long_beach_train_2, y_long_beach_test_2)

print("SWITZERLAND")
run(GaussianNB(), X_switzerland_train_1, X_switzerland_test_1, y_switzerland_train_1, y_switzerland_test_1,
   X_switzerland_train_2, X_switzerland_test_2, y_switzerland_train_2, y_switzerland_test_2)

print("HUNGARY")
run(GaussianNB(), X_hungary_train_1, X_hungary_test_1, y_hungary_train_1, y_hungary_test_1,
   X_hungary_train_2, X_hungary_test_2, y_hungary_train_2, y_hungary_test_2)

# K-nearest neighbor Model
# This model performs rather disappointingly on all of the datasets. It is close to a random classifier for the 
# CLEVELAND, LONG BEACH AND SWITZERLAND datasets. Its AUC of 0.71 on the Hungarian dataset is not outstanding, either.
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

print("CLEVELAND")
run(KNeighborsClassifier(), X_cleveland_train_1, X_cleveland_test_1, y_cleveland_train_1, y_cleveland_test_1,
   X_cleveland_train_2, X_cleveland_test_2, y_cleveland_train_2, y_cleveland_test_2)

print("LONG BEACH")
run(KNeighborsClassifier(), X_long_beach_train_1, X_long_beach_test_1, y_long_beach_train_1, y_long_beach_test_1,
   X_long_beach_train_2, X_long_beach_test_2, y_long_beach_train_2, y_long_beach_test_2)

print("SWITZERLAND")
run(KNeighborsClassifier(), X_switzerland_train_1, X_switzerland_test_1, y_switzerland_train_1, y_switzerland_test_1,
   X_switzerland_train_2, X_switzerland_test_2, y_switzerland_train_2, y_switzerland_test_2)

print("HUNGARY")
run(KNeighborsClassifier(), X_hungary_train_1, X_hungary_test_1, y_hungary_train_1, y_hungary_test_1,
   X_hungary_train_2, X_hungary_test_2, y_hungary_train_2, y_hungary_test_2)

# DecisionTree Model
# This model performs rather poorly on all 4 datasets. It is a worse-than-random classifier on the CLEVELAND dataset,
# while performs slightly better than a random classifier on the other datasets. 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

print("CLEVELAND")
run(DecisionTreeClassifier(), X_cleveland_train_1, X_cleveland_test_1, y_cleveland_train_1, y_cleveland_test_1,
   X_cleveland_train_2, X_cleveland_test_2, y_cleveland_train_2, y_cleveland_test_2)

print("LONG BEACH")
run(DecisionTreeClassifier(), X_long_beach_train_1, X_long_beach_test_1, y_long_beach_train_1, y_long_beach_test_1,
   X_long_beach_train_2, X_long_beach_test_2, y_long_beach_train_2, y_long_beach_test_2)

print("SWITZERLAND")
run(DecisionTreeClassifier(), X_switzerland_train_1, X_switzerland_test_1, y_switzerland_train_1, y_switzerland_test_1,
   X_switzerland_train_2, X_switzerland_test_2, y_switzerland_train_2, y_switzerland_test_2)

print("HUNGARY")
run(DecisionTreeClassifier(), X_hungary_train_1, X_hungary_test_1, y_hungary_train_1, y_hungary_test_1,
   X_hungary_train_2, X_hungary_test_2, y_hungary_train_2, y_hungary_test_2)

# Support Vector Machine Model
# This model performs extremely poorly on the LONG BEACH dataset. It is close to a random classifier on the 
# CLEVELAND dataset, while does slightly better on the Hungary and the Switzerland datasets.
from sklearn import metrics
from sklearn.svm import SVC

print("CLEVELAND")
run(SVC(probability = True), X_cleveland_train_1, X_cleveland_test_1, y_cleveland_train_1, y_cleveland_test_1,
   X_cleveland_train_2, X_cleveland_test_2, y_cleveland_train_2, y_cleveland_test_2)

print("LONG BEACH")
run(SVC(probability = True), X_long_beach_train_1, X_long_beach_test_1, y_long_beach_train_1, y_long_beach_test_1,
   X_long_beach_train_2, X_long_beach_test_2, y_long_beach_train_2, y_long_beach_test_2)

print("SWITZERLAND")
run(SVC(probability = True), X_switzerland_train_1, X_switzerland_test_1, y_switzerland_train_1, y_switzerland_test_1,
   X_switzerland_train_2, X_switzerland_test_2, y_switzerland_train_2, y_switzerland_test_2)

print("HUNGARY")
run(SVC(probability = True), X_hungary_train_1, X_hungary_test_1, y_hungary_train_1, y_hungary_test_1,
   X_hungary_train_2, X_hungary_test_2, y_hungary_train_2, y_hungary_test_2)

# My model selections:
# Cleveland: LogisticRegression
# Long Beach: LogisticRegression
# Switzerland: DecisionTree
# Hungary: Gaussian



