# Problem 1 is about comparing the performance of various cross-validation techniques using a Linear Regression model.
# It seems that using the hold-out and the 10-fold CV methods, the DecisionTree Regression model badly underfits the data, while the
# Linear Regression and the SVM regressor models perform similarly. The leave-one-out method has not finished running
# at the time of the submission. (The printed result is from an incorrect previous trial)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Read in data
data = pd.read_excel('Book1.xlsx')
data = np.asarray(data)
data_x = data[:,1]
data_y = data[:,2]

# HOLD-OUT technique on Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=15)
lr = LinearRegression()
dt = DecisionTreeRegressor()
sv = SVR(kernel="linear", gamma = "auto")
lr.fit(X_train.reshape(-1, 1), y_train)
dt.fit(X_train.reshape(-1, 1), y_train)
sv.fit(X_train.reshape(-1, 1), y_train)

pred_lr = lr.predict(X_test.reshape(-1, 1))
pred_dt = dt.predict(X_test.reshape(-1, 1))
pred_sv = sv.predict(X_test.reshape(-1, 1))

mse_holdout_lr = mean_squared_error(y_test, pred_lr)
mse_holdout_dt = mean_squared_error(y_test, pred_dt)
mse_holdout_sv = mean_squared_error(y_test, pred_sv)

print('Hold-out technique Linear Regression MSE: {:f}'.format(mse_holdout_lr))
print('Hold-out technique DecisionTree MSE: {:f}'.format(mse_holdout_dt))
print('Hold-out technique SVM MSE: {:f}'.format(mse_holdout_sv))

# 10-fold Cross-Validation technique on Linear Regression Model
mse_cv_lr = []
mse_cv_dt = []
mse_cv_sv = []
lr = LinearRegression()
dt = DecisionTreeRegressor()
sv = SVR(kernel="linear", gamma="auto")
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(data_x):
    X_train, X_test = data_x[train_index], data_x[test_index]
    y_train, y_test = data_y[train_index], data_y[test_index]
    lr.fit(X_train.reshape(-1, 1), y_train)
    dt.fit(X_train.reshape(-1, 1), y_train)
    sv.fit(X_train.reshape(-1, 1), y_train)

    pred_lr = lr.predict(X_test.reshape(-1, 1))
    pred_dt = dt.predict(X_test.reshape(-1, 1))
    pred_sv = sv.predict(X_test.reshape(-1, 1))

    mse_lr = mean_squared_error(y_test, pred_lr)
    mse_cv_lr.append(mse_lr)
    mse_dt = mean_squared_error(y_test, pred_dt)
    mse_cv_dt.append(mse_dt)
    mse_sv = mean_squared_error(y_test, pred_sv)
    mse_cv_sv.append(mse_sv)

mse_kfold_lr = sum(mse_cv_lr) / len(mse_cv_lr)
mse_kfold_dt = sum(mse_cv_dt) / len(mse_cv_dt)
mse_kfold_sv = sum(mse_cv_sv) / len(mse_cv_sv)
print('10-fold Cross-validation technique average MSE (Lin. Reg.): {:f}'.format(mse_kfold_lr))
print('10-fold Cross-validation technique average MSE (Dec. Tree): {:f}'.format(mse_kfold_dt))
print('10-fold Cross-validation technique average MSE (SVM): {:f}'.format(mse_kfold_sv))

# Leave-one-out Cross-Validation technique on Linear Regression Model
mse_cv_lr = []
mse_cv_dt = []
mse_cv_sv = []
lr = LinearRegression()
dt = DecisionTreeRegressor()
sv = SVR(kernel="linear", gamma="auto")
kf = KFold(n_splits=len(data_x))
for train_index, test_index in kf.split(data_x):
    X_train, X_test = data_x[train_index], data_x[test_index]
    y_train, y_test = data_y[train_index], data_y[test_index]
    lr.fit(X_train.reshape(-1, 1), y_train)
    dt.fit(X_train.reshape(-1, 1), y_train)
    sv.fit(X_train.reshape(-1, 1), y_train)

    pred_lr = lr.predict(X_test.reshape(-1, 1))
    pred_dt = dt.predict(X_test.reshape(-1, 1))
    pred_sv = sv.predict(X_test.reshape(-1, 1))

    mse_lr = mean_squared_error(y_test, pred_lr)
    mse_cv_lr.append(mse_lr)
    mse_dt = mean_squared_error(y_test, pred_dt)
    mse_cv_dt.append(mse_dt)
    mse_sv = mean_squared_error(y_test, pred_sv)
    mse_cv_sv.append(mse_sv)

mse_kfold_lr = sum(mse_cv_lr) / len(mse_cv_lr)
mse_kfold_dt = sum(mse_cv_dt) / len(mse_cv_dt)
mse_kfold_sv = sum(mse_cv_sv) / len(mse_cv_sv)
print('Leave One out Cross-validation technique average MSE (Lin. Reg.): {:f}'.format(mse_kfold_lr))
print('Leave One out Cross-validation technique average MSE (Dec. Tree): {:f}'.format(mse_kfold_dt))
print('Leave One out Cross-validation technique average MSE (SVM): {:f}'.format(mse_kfold_sv))


# Problem 2 is about comparing the performance of various models, given the same cross-validation technique(10-fold).
# Unlike in Problem 1, in this case, we can observe major differences in performance between the various models.
import os
csv_path = os.path.join( os.path.join("datasets", "housing"), "housing.csv")
housing = pd.read_csv(csv_path)

# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# Using stratify split
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing_labels = strat_train_set["median_house_value"].copy()
housing = strat_train_set.drop("median_house_value", axis=1)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_num = housing.select_dtypes(include=[np.number])

from sklearn.preprocessing import Imputer
from future_encoders import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from future_encoders import ColumnTransformer

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# 10-fold Cross-Validation Technique on Support Vector Machine Regressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

svr = SVR(kernel="linear", gamma = "auto")
svr.fit(housing_prepared, housing_labels)
scores = cross_val_score(svr, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
svr_rmse_scores = np.sqrt(-scores)
print("Root mean squared error of support vector machine regressor model: %.2f" % np.average(svr_rmse_scores))

# 10-fold Cross-Validation Technique on Decision Tree Model
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("Root mean squared error of decision tree model: %.2f" % np.average(tree_rmse_scores))

# 10-fold Cross-Validation Technique on Linear Regression Model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Root mean squared error of linear model: %.2f" %np.average(lin_rmse_scores))

# CONCLUSION

# As we can see from the above values, the Support Vector Machine Regressor model has the highest Root Mean
# Squared Error value on the training set. This is clearly an indication of the model badly underfitting the
# training data. Based on the above results, it seems like the Linear Regression model is doing the best job
# at fitting the data. 