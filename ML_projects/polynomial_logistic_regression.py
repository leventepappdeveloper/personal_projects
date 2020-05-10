# Levente Papp
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# generate data
X = 2 * np.random.rand(100, 1)
y = 5*np.sin(6*X) + 2*np.cos(X) + np.random.randn(100, 1)

# fit polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

n = 2
poly_features = PolynomialFeatures(degree = n, include_bias = False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown

from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=5, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 80, 0, 3])

# LEVENTE PAPP, 10/5/2019
# As we can see, the best degree of the parameter is 17 in this case, because that is where it reaches the lowest
# RMSE value on the validation data.
from sklearn.model_selection import GridSearchCV

parameters = {'poly_features__degree': np.asarray(range(20))+1}
clf = GridSearchCV(polynomial_regression, parameters, cv=5, scoring='neg_mean_squared_error')
clf.fit(X, y)
print(clf.best_params_)

# LEVENTE PAPP, 10/5/2019
# Create data set and fit logistic regression model. As we can see, the model's prediction for someone
# who studied 4.2 hours would be that they PASS (1) the exam.
import pandas as pd

data = [[0.5, 0], [0.5, 0], [0.5, 0], [0.5, 0], [0.5, 0], [0.5, 1], [1, 0], [1, 0], [1, 0], [1.5, 0],
       [1.5, 0], [1.5, 0], [1.5, 0], [1.5, 0], [1.5, 0], [1.5, 1], [1.5, 1], [2, 0], [2, 0], [2.5, 0],
       [2.5, 0], [2.5, 0], [3, 0], [3, 0], [3, 1], [3.5, 0], [3.5, 0], [3.5, 1], [4, 0], [4, 0],
       [4.5, 1], [4.5, 1], [4.5, 1], [4.5, 1], [5, 0], [5, 1], [5, 1], [5, 1], [5, 1], [5, 1],
       [5.5, 1], [5.5, 1], [6, 1], [6, 1], [6.5, 0], [7, 1]]

data_frame = pd.DataFrame(data, columns = ['Hours', 'Passed'])

X = data_frame['Hours'].to_numpy().reshape(-1, 1)
y = np.array(data_frame['Passed'])

from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(random_state=0, solver='lbfgs')
logistic_regression.fit(X, y)
logistic_regression.predict([[4.2]])



