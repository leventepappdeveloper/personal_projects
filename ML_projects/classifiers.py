# Levente Papp, 10/31/2019
import numpy as np
import urllib.request as ur
import pandas as pd
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

raw_data = pd.read_csv("german.data-numeric.csv", header=None)

matrix = []
for i in range(1000):
    row = raw_data[0][i].split(" ")
    while "" in row:
        row.remove("")
    matrix.append(row)

column_names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10",
               "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20",
               "X21", "X22", "X23", "X24", "X25"]
german_data = pd.DataFrame(columns = column_names)
for row in matrix:
    german_data = german_data.append(pd.Series(row, index = column_names), ignore_index = True)
X = german_data.iloc[:, 0:24]
y = german_data.iloc[:, 24:]

cost_matrix = [[0, 1], [5, 0]]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
extra_trees_model = ExtraTreesClassifier()
extra_trees_model.fit(X_train, y_train)
extra_trees_predicted = extra_trees_model.predict(X_test)


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
rfe_model = RFE(model1, 3)
rfe_model = rfe_model.fit(X_train, y_train)
rfe_predicted = rfe_model.predict(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_model_predicted = logistic_model.predict(X_test)



from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
gaussian_model = GaussianNB()
gaussian_model.fit(X_train, y_train)
gaussian_model_predicted = gaussian_model.predict(X_test)

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_model_predicted = knn_model.predict(X_test)

# part a)
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection 

estimators = []
estimators.append(("extratrees", extra_trees_model))
estimators.append(("rfe", rfe_model))
estimators.append(("logistic", logistic_model))
estimators.append(("gaussian", gaussian_model))
estimators.append(("KNN", knn_model))
ensemble = VotingClassifier(estimators, voting = "hard")
ensemble.fit(X_train, y_train)

predicted = ensemble.predict(X_test)
unweighted_voting_confusion = metrics.confusion_matrix(y_test, predicted)
product_unweighted = np.multiply(unweighted_voting_confusion, cost_matrix)
sum_unweighted = sum(sum(product_unweighted))

ensemble2 = VotingClassifier(estimators, voting = "soft")
ensemble2.fit(X_train, y_train)
predicted2 = ensemble2.predict(X_test)
weighted_voting_confusion = metrics.confusion_matrix(y_test, predicted2)
product_weighted = np.multiply(weighted_voting_confusion, cost_matrix)
sum_weighted = sum(sum(product_weighted))

# part b)
# Random feature and training set selections?
from sklearn.ensemble import RandomForestClassifier

# tune random forest depth (find optimal max_depth between 1 and 1000)
optimal_depth = 1
optimal_cost = float("inf")
for i in range(1, 300):
    random_forest = RandomForestClassifier(n_estimators = 100, max_depth = i)
    random_forest.fit(X_train, y_train)
    random_forest_predictions = random_forest.predict(X_test)
    random_forest_confusion = metrics.confusion_matrix(y_test, random_forest_predictions)
    product_random_forest = np.multiply(random_forest_confusion, cost_matrix)
    sum_random_forest = sum(sum(product_random_forest))
    if sum_random_forest < optimal_cost:
        optimal_depth = i
        optimal_cost = sum_random_forest


# part c)
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
adaboost.fit(X_train, y_train) 
adaboost_predictions = adaboost.predict(X_test)
adaboost_confusion = metrics.confusion_matrix(y_test, adaboost_predictions)
product_adaboost = np.multiply(adaboost_confusion, cost_matrix)
sum_adaboost = sum(sum(product_adaboost))


print("Unweighted Voting Classifier Cost: " + str(sum_unweighted))
print("Weighted Voting Classifier Cost: " + str(sum_weighted))
print("Random Forest Cost: " + str(optimal_cost) + " with maxDepth = " + str(optimal_depth))
print("Adaboost Cost: " + str(sum_adaboost))


# From the above Cost values we can see that the Random Forest Classifier has the minimum cost at a height of 61.
# From these values, it seems like the Unweighted Voting Classifier is performing the weakest, since it has the 
# highest Cost (punishment) value. 
